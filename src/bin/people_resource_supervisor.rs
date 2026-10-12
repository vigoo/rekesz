// Linux-only release-validation tooling: it samples process-group RSS from
// /proc and relies on prctl(2) subreaper semantics, neither of which exist on
// other platforms. The release-validation workflow only runs on Linux.
use std::process::ExitCode;

#[cfg(target_os = "linux")]
use std::{
    env,
    ffi::CString,
    fs::File,
    io::{self, Read, Write},
    os::fd::FromRawFd,
    sync::atomic::{AtomicI32, Ordering},
    thread,
    time::{Duration, Instant},
};

fn main() -> ExitCode {
    #[cfg(not(target_os = "linux"))]
    {
        eprintln!("people resource supervisor: only supported on Linux");
        return ExitCode::FAILURE;
    }

    #[cfg(target_os = "linux")]
    match run() {
        Ok(code) => ExitCode::from(code.clamp(0, 255) as u8),
        Err(error) => {
            eprintln!("people resource supervisor: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(target_os = "linux")]
static FORWARDED_SIGNAL: AtomicI32 = AtomicI32::new(0);

#[cfg(target_os = "linux")]
extern "C" fn record_signal(signal: libc::c_int) {
    FORWARDED_SIGNAL.store(signal, Ordering::Release);
}

#[cfg(target_os = "linux")]
fn usage_seconds(value: libc::timeval) -> f64 {
    value.tv_sec as f64 + value.tv_usec as f64 / 1_000_000.0
}

#[cfg(target_os = "linux")]
fn process_group_rss_kib(group: libc::pid_t) -> io::Result<i64> {
    if env::var_os("PEOPLE_SUPERVISOR_FAIL_SAMPLER").is_some() {
        return Err(io::Error::other("injected process-group sampler failure"));
    }
    let mut total = 0_i64;
    for entry in std::fs::read_dir("/proc")? {
        let entry = entry?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if !name.bytes().all(|byte| byte.is_ascii_digit()) {
            continue;
        }
        let Ok(stat) = std::fs::read_to_string(entry.path().join("stat")) else {
            continue;
        };
        let Some(after_name) = stat.rsplit_once(") ").map(|(_, rest)| rest) else {
            continue;
        };
        let fields = after_name.split_ascii_whitespace().collect::<Vec<_>>();
        // Fields after comm begin at proc field 3: pgrp is field 5.
        if fields.get(2).and_then(|value| value.parse::<i32>().ok()) != Some(group) {
            continue;
        }
        let Ok(status) = std::fs::read_to_string(entry.path().join("status")) else {
            continue;
        };
        if let Some(rss) = status.lines().find_map(|line| {
            line.strip_prefix("VmRSS:")
                .and_then(|value| value.split_ascii_whitespace().next())
                .and_then(|value| value.parse::<i64>().ok())
        }) {
            total = total.saturating_add(rss);
        }
    }
    Ok(total)
}

#[cfg(target_os = "linux")]
unsafe fn set_signal_mask(how: libc::c_int, mask: &libc::sigset_t) -> io::Result<()> {
    if unsafe { libc::sigprocmask(how, mask, std::ptr::null_mut()) } == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(target_os = "linux")]
fn run() -> Result<i32, Box<dyn std::error::Error>> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    let separator = arguments
        .iter()
        .position(|value| value == "--")
        .ok_or("missing --")?;
    if separator != 4 || arguments.len() <= 5 {
        return Err(
            "usage: supervisor RESOURCES LOG MAX_RSS_KIB MAX_CPU_PERCENT -- COMMAND ...".into(),
        );
    }
    let resources = arguments[0].clone();
    let log = arguments[1].clone();
    let max_rss = arguments[2].parse::<i64>()?;
    let max_cpu = arguments[3].parse::<f64>()?;
    let command = arguments[5..]
        .iter()
        .map(|value| CString::new(value.as_str()))
        .collect::<Result<Vec<_>, _>>()?;
    let mut blocked = unsafe { std::mem::zeroed::<libc::sigset_t>() };
    unsafe {
        libc::sigemptyset(&mut blocked);
        libc::sigaddset(&mut blocked, libc::SIGINT);
        libc::sigaddset(&mut blocked, libc::SIGTERM);
        libc::sigaddset(&mut blocked, libc::SIGHUP);
        set_signal_mask(libc::SIG_BLOCK, &blocked)?;
    }
    for signal in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP] {
        unsafe { libc::signal(signal, record_signal as *const () as libc::sighandler_t) };
    }
    let mut pipe = [0; 2];
    if unsafe { libc::pipe(pipe.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error().into());
    }
    let mut readiness = [0; 2];
    if unsafe { libc::pipe(readiness.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error().into());
    }
    if unsafe { libc::prctl(libc::PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) } != 0 {
        return Err(io::Error::last_os_error().into());
    }
    let child = unsafe { libc::fork() };
    if child < 0 {
        return Err(io::Error::last_os_error().into());
    }
    if child == 0 {
        unsafe {
            libc::close(pipe[0]);
            libc::close(readiness[0]);
            if libc::setsid() < 0 {
                libc::_exit(126);
            }
            libc::signal(libc::SIGINT, libc::SIG_DFL);
            libc::signal(libc::SIGTERM, libc::SIG_DFL);
            libc::signal(libc::SIGHUP, libc::SIG_DFL);
            if set_signal_mask(libc::SIG_UNBLOCK, &blocked).is_err() {
                libc::_exit(126);
            }
            if let Some(pid_file) = env::var_os("PEOPLE_SUPERVISOR_TEST_CHILD_PID_FILE") {
                let _ = std::fs::write(pid_file, format!("{}", libc::getpid()));
            }
            if let Some(marker) = env::var_os("PEOPLE_SUPERVISOR_TEST_PRE_READY_MARKER") {
                let _ = std::fs::write(marker, b"held");
                if let Some(release) = env::var_os("PEOPLE_SUPERVISOR_TEST_PRE_READY_RELEASE") {
                    while !std::path::Path::new(&release).exists() {
                        libc::usleep(1_000);
                    }
                }
            }
            if libc::write(readiness[1], b"R".as_ptr().cast(), 1) != 1 {
                libc::_exit(126);
            }
            libc::close(readiness[1]);
            libc::dup2(pipe[1], libc::STDOUT_FILENO);
            libc::dup2(pipe[1], libc::STDERR_FILENO);
            libc::close(pipe[1]);
            let mut pointers = command
                .iter()
                .map(|value| value.as_ptr())
                .collect::<Vec<_>>();
            pointers.push(std::ptr::null());
            libc::execvp(pointers[0], pointers.as_ptr());
            libc::_exit(127);
        }
    }
    unsafe {
        libc::close(pipe[1]);
        libc::close(readiness[1]);
    }
    let mut ready = 0_u8;
    let ready_count = unsafe { libc::read(readiness[0], (&mut ready as *mut u8).cast(), 1) };
    unsafe { libc::close(readiness[0]) };
    if ready_count != 1 || ready != b'R' {
        unsafe {
            libc::kill(-child, libc::SIGKILL);
            libc::kill(child, libc::SIGKILL);
            libc::close(pipe[0]);
            libc::waitpid(child, std::ptr::null_mut(), 0);
        }
        return Err("child exited before process-group readiness".into());
    }
    unsafe { set_signal_mask(libc::SIG_UNBLOCK, &blocked)? };
    let logger = thread::spawn(move || -> io::Result<()> {
        let mut source = unsafe { File::from_raw_fd(pipe[0]) };
        let mut target = File::create(log)?;
        let mut stdout = io::stdout().lock();
        let mut buffer = [0_u8; 16 * 1024];
        loop {
            let count = source.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            target.write_all(&buffer[..count])?;
            stdout.write_all(&buffer[..count])?;
        }
        target.flush()?;
        stdout.flush()
    });
    let started = Instant::now();
    let mut leader_status = None;
    let mut max_rss_kib = 0_i64;
    let mut sampled_group_peak_rss_kib = 0_i64;
    let mut cpu_seconds = 0.0_f64;
    let mut forwarded = 0;
    let mut sampler_error = None;
    loop {
        match process_group_rss_kib(child) {
            Ok(rss) => sampled_group_peak_rss_kib = sampled_group_peak_rss_kib.max(rss),
            Err(error) if sampler_error.is_none() => {
                sampler_error = Some(error.to_string());
                unsafe { libc::kill(-child, libc::SIGKILL) };
            }
            Err(_) => {}
        }
        let signal = FORWARDED_SIGNAL.load(Ordering::Acquire);
        if signal != 0 && signal != forwarded {
            let result = unsafe { libc::kill(-child, signal) };
            if result == 0 || leader_status.is_some() {
                forwarded = signal;
            } else if io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH) {
                unsafe { libc::kill(-child, libc::SIGKILL) };
            }
        }
        let mut status = 0;
        let mut usage = unsafe { std::mem::zeroed::<libc::rusage>() };
        let waited = unsafe { libc::wait4(-1, &mut status, libc::WNOHANG, &mut usage) };
        if waited > 0 {
            max_rss_kib = max_rss_kib.max(usage.ru_maxrss);
            cpu_seconds += usage_seconds(usage.ru_utime) + usage_seconds(usage.ru_stime);
            if waited == child {
                leader_status = Some(status);
            }
            continue;
        }
        if waited < 0 {
            let error = io::Error::last_os_error();
            if error.raw_os_error() == Some(libc::ECHILD) {
                break;
            }
            if error.raw_os_error() != Some(libc::EINTR) {
                return Err(error.into());
            }
        }
        thread::sleep(Duration::from_millis(5));
    }
    logger.join().map_err(|_| "logger panicked")??;
    let wall = started.elapsed().as_secs_f64().max(0.001);
    let cpu_percent = cpu_seconds / wall * 100.0;
    let enforced_peak_rss_kib = max_rss_kib.max(sampled_group_peak_rss_kib);
    let mut artifact = File::create(resources)?;
    writeln!(artifact, "wall_seconds={wall:.3}")?;
    writeln!(
        artifact,
        "sampled_group_peak_rss_kib={sampled_group_peak_rss_kib}"
    )?;
    writeln!(artifact, "max_individual_peak_rss_kib={max_rss_kib}")?;
    writeln!(artifact, "max_rss_kib={enforced_peak_rss_kib}")?;
    writeln!(artifact, "cpu_percent={cpu_percent:.1}%")?;
    if let Some(error) = &sampler_error {
        writeln!(artifact, "sampler_error={error}")?;
    }
    artifact.flush()?;
    if let Some(error) = sampler_error {
        return Err(format!("process-group RSS sampler failed: {error}").into());
    }
    if enforced_peak_rss_kib > max_rss || cpu_percent > max_cpu {
        eprintln!(
            "resource budget exceeded: rss={enforced_peak_rss_kib}/{max_rss} KiB cpu={cpu_percent:.1}/{max_cpu:.1}%"
        );
        return Ok(1);
    }
    let status = leader_status.ok_or("leader status was not reaped")?;
    if libc::WIFEXITED(status) {
        Ok(libc::WEXITSTATUS(status))
    } else if libc::WIFSIGNALED(status) {
        Ok(128 + libc::WTERMSIG(status))
    } else {
        Ok(1)
    }
}
