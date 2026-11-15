# Push-scan test data

- `announcement-capture-2026-11-14.log` — raw log of the live listener capture
  (L3160, 192.168.50.67): the `02 06` NetScanMonitor announcements the panel's
  Scan → Computer menu emits (seq byte 11 increments per burst, 3× each).
- `l3160-*.bin` — INIT1/INIT2 replies captured by `scanner-probe push-probe`
  against the L3160 (2026-11-14): welcome (esci2-plain classification), INFO,
  CAPA, RESA bodies + their 64-byte META frames, and the probe's traffic log.
- `et4800-flatbed-jpg-{printer,host}.bin` — both directions of a real ET-4800
  flatbed JPEG session (committed capture from epson2paperless,
  `tools/pcap-extract/captures/et-4800/flatbed-jpg.jsonl`, MIT). The
  ET-4800's CAPA shape matches the L3160's; the session replay test drives
  the state machine over the printer stream and asserts the host stream is
  byte-identical.
- `golden-para-flatbed-{300,600}dpi.bin` — the composed PARA bodies the
  reference's own composer produces for the ET-4800 dialect inputs
  (flatbed, jpg, A4 window, quality 90); the 300-DPI golden is
  byte-identical to the PARA inside the recorded session above.
