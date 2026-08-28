use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
};

use crate::rendering::{SourceIdentity, decoded::DecodedSource};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(in crate::rendering) struct DecodedSourceKey {
    pub(in crate::rendering) path: PathBuf,
    pub(in crate::rendering) identity: SourceIdentity,
}

enum DecodedSourceEntry {
    Loading,
    Ready(Arc<DecodedSource>),
}

/// RAW-only reuse of decoded linear sources. Callers consult the render plan
/// before entering this cache, keeping variant-sensitive standard decodes out.
pub(in crate::rendering) struct DecodedSourceCache {
    entries: Mutex<HashMap<DecodedSourceKey, DecodedSourceEntry>>,
    order: Mutex<VecDeque<DecodedSourceKey>>,
    wake: Condvar,
    capacity: AtomicUsize,
    hits: AtomicU64,
}

impl DecodedSourceCache {
    pub(in crate::rendering) fn new(capacity: usize) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            order: Mutex::new(VecDeque::new()),
            wake: Condvar::new(),
            capacity: AtomicUsize::new(capacity.max(1)),
            hits: AtomicU64::new(0),
        }
    }

    pub(in crate::rendering) fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    pub(in crate::rendering) fn set_capacity(&self, capacity: usize) {
        let capacity = capacity.max(1);
        self.capacity.store(capacity, Ordering::Relaxed);
        let mut entries = self.entries.lock().unwrap();
        let mut order = self.order.lock().unwrap();
        while order.len() > capacity {
            if let Some(oldest) = order.pop_front() {
                entries.remove(&oldest);
            }
        }
    }

    pub(in crate::rendering) fn get_or_decode(
        &self,
        key: DecodedSourceKey,
        cancelled: &AtomicBool,
        decode: impl FnOnce() -> image::ImageResult<Arc<DecodedSource>>,
    ) -> image::ImageResult<Arc<DecodedSource>> {
        let mut decode = Some(decode);
        loop {
            let mut entries = self.entries.lock().unwrap();
            match entries.get(&key) {
                Some(DecodedSourceEntry::Ready(source)) => {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    return Ok(source.clone());
                }
                Some(DecodedSourceEntry::Loading) => {
                    if cancelled.load(Ordering::Acquire) {
                        return Err(cancelled_image_error());
                    }
                    drop(
                        self.wake
                            .wait_timeout(entries, std::time::Duration::from_millis(20))
                            .unwrap(),
                    );
                }
                None => {
                    entries.insert(key.clone(), DecodedSourceEntry::Loading);
                    drop(entries);
                    let result = decode.take().expect("decode closure used once")();
                    let mut entries = self.entries.lock().unwrap();
                    match &result {
                        Ok(source) => {
                            entries.insert(key.clone(), DecodedSourceEntry::Ready(source.clone()));
                            let mut order = self.order.lock().unwrap();
                            order.retain(|candidate| candidate != &key);
                            order.push_back(key.clone());
                            while order.len() > self.capacity.load(Ordering::Relaxed) {
                                if let Some(oldest) = order.pop_front()
                                    && oldest != key
                                {
                                    entries.remove(&oldest);
                                }
                            }
                        }
                        Err(_) => {
                            entries.remove(&key);
                        }
                    }
                    drop(entries);
                    self.wake.notify_all();
                    return result;
                }
            }
        }
    }
}

fn cancelled_image_error() -> image::ImageError {
    image::ImageError::Decoding(image::error::DecodingError::new(
        image::error::ImageFormatHint::Unknown,
        "render cancelled",
    ))
}

#[cfg(test)]
mod tests;
