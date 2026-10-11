# Single-emulator test without KVM

Run `sbatch scripts/orion/android-soft.sbatch` from the repository root after
syncing. The job requests **testqueue, one GPU node, 8 CPUs, 24 GiB, 90 minutes**,
account `prjsnnn0000269`. It uses that GPU node's CPU, forces `-accel off`, and
uses software graphics (`-gpu swiftshader`). It does not load a model or connect
to another compute node. A future model can run alongside the emulator only
after this diagnostic succeeds and combined resource usage is checked.

The first job installs Java 17, Android SDK command-line tools and the stable
emulator/platform-tools plus Android 13/API 33 Google APIs x86_64 image. SDK
installation accepts the requested packages' licenses. Downloads and persistent
files go under `/scratch/prjsnnn0000269/Students/shuming/cjk/android-soft-probe`.
The job uses its own SDK and Java; it does not alter the existing game/model
venvs. Java and command-line-tool downloads are checksum verified; the other
packages are installed by sdkmanager from Google's repositories. Actual package
revisions are saved in the JSON report; completed SDK setup is reused without
automatic upgrades. A failed setup is distinct from a failed emulator test.

The test creates a fresh Pixel 6 AVD for each job, uses a private ADB server and
unused emulator/gRPC ports, and waits up to **20 minutes** for Android boot. It
then unlocks/provisions this diagnostic AVD, opens Settings, saves a screenshot
and UI tree, taps a settings entry, verifies subpage content and saves another
screenshot. A 30-minute outer probe limit also bounds interaction/cleanup. Only
process groups created by this test are terminated; shared ADB servers and other
emulators are not stopped. SDK/AVD installation never takes place on login nodes.

Inspect the repository's `android-soft-JOBID.out`. Once the Python probe starts,
its detailed outputs are in `android-soft-probe/jobs/JOBID/`:

- `report.json`: status, failing stage, command, revisions, timings and ports.
- `emulator.log`, `adb.log`: launch/transport failures or missing system libraries.
- `before.png`, `after.png`, `before.xml`, `after.xml`: available after those steps.

`ANDROID_SOFTWARE_PROBE_OK` confirms this basic diagnostic passed. A
`boot_timeout` means this setup did not boot within the limit, not that all
software emulation is impossible. An immediate emulator exit may instead expose
unsupported acceleration settings or missing host libraries; inspect the log.

This is **not yet an AndroidWorld task evaluation**: benchmark apps, accessibility
forwarding, rewards and model interaction have not been installed/tested. The
CPU workload and screenshots establish whether proceeding is practical. For
formal experiments, record/freeze SDK and image revisions and measure task
throughput rather than extrapolating from this one Settings interaction.

References: [AndroidWorld AVD requirements](https://github.com/google-research/android_world#installation),
[emulator acceleration](https://developer.android.com/studio/run/emulator-acceleration),
[Google SDK metadata](https://dl.google.com/android/repository/repository2-1.xml).
