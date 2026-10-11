#!/bin/bash -l
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --job-name=android-soft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=android-soft-%j.out

# A single software emulator on the CPU of a GPU node. No model is loaded.
set -euo pipefail
ulimit -c 0
ROOT=/scratch/prjsnnn0000269/Students/shuming/cjk
REPO="${ORION_REPO:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
BASE="$ROOT/android-soft-probe"
JOB_DIR="$BASE/jobs/$SLURM_JOB_ID"
export ANDROID_HOME="$BASE/sdk"
export ANDROID_SDK_ROOT="$ANDROID_HOME"
# Keep modern and legacy preference locations consistent, isolated per job.
export ANDROID_SDK_HOME="$JOB_DIR"
export ANDROID_USER_HOME="$ANDROID_SDK_HOME/.android"
export ANDROID_EMULATOR_HOME="$ANDROID_USER_HOME"
export ANDROID_AVD_HOME="$ANDROID_USER_HOME/avd"
# Short node-local paths avoid Unix socket path-length limits. Persistent SDK,
# AVDs, reports and screenshots remain in Scratch.
export TMPDIR="/tmp/cjk-android-soft-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$ROOT/.cache"
export JAVA_HOME="$BASE/jdk-17.0.16+8"
export PATH="$JAVA_HOME/bin:$PATH"
unset PYTHONHOME PYTHONPATH JAVA_TOOL_OPTIONS JDK_JAVA_OPTIONS
mkdir -p "$BASE/downloads" "$ANDROID_HOME/cmdline-tools" "$ANDROID_USER_HOME" "$ANDROID_AVD_HOME" "$TMPDIR" "$ROOT/logs"
cd "$REPO"
echo "host=$(hostname) partition=$SLURM_JOB_PARTITION commit=$(git rev-parse HEAD)"
echo "scope=one emulator, -accel off, software graphics, no model inference"
PHASE=prerequisites
on_failure() {
    local code="$1" line="$2"
    printf '{"status":"failed","stage":"%s","exit_code":%s,"line":%s}\n' \
      "$PHASE" "$code" "$line" > "$JOB_DIR/setup-status.json"
    echo "ANDROID_SOFT_FAILED: stage=$PHASE exit=$code line=$line" >&2
    echo "Details: $JOB_DIR (AVD creation: avd-create.log; emulator test: report.json)" >&2
}
trap 'on_failure "$?" "$LINENO"' ERR
for tool in curl unzip tar timeout flock tee; do command -v "$tool" >/dev/null; done
/usr/bin/python3 -I -B -c 'import sys; assert sys.version_info >= (3, 8); print(sys.version)'

# Serialize the diagnostic, including setup, across our own jobs.
exec 9>"$BASE/probe.lock"
flock -n 9 || { echo 'Another Android software probe is already running.' >&2; exit 1; }
PHASE=java_setup
if [[ ! -x "$JAVA_HOME/bin/java" ]]; then
    JDK=OpenJDK17U-jdk_x64_linux_hotspot_17.0.16_8.tar.gz
    curl --fail --location --retry 3 --max-time 600 \
      "https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.16%2B8/$JDK" \
      -o "$BASE/downloads/$JDK"
    (cd "$BASE/downloads" && printf '%s  %s\n' \
      '166774efcf0f722f2ee18eba0039de2d685b350ee14d7b69e6f83437dafd2af1' "$JDK" | sha256sum -c -)
    STAGE="$(mktemp -d "$BASE/java-stage.XXXXXX")"
    tar --no-same-owner -xzf "$BASE/downloads/$JDK" -C "$STAGE"
    [[ ! -e "$JAVA_HOME" ]]
    mv "$STAGE/jdk-17.0.16+8" "$JAVA_HOME"
    rmdir "$STAGE"
fi
SDKMANAGER="$ANDROID_HOME/cmdline-tools/12.0/bin/sdkmanager"
PHASE=command_line_tools_setup
if [[ ! -x "$SDKMANAGER" ]]; then
    ZIP=commandlinetools-linux-11076708_latest.zip
    curl --fail --location --retry 3 --max-time 600 \
      "https://dl.google.com/android/repository/$ZIP" -o "$BASE/downloads/$ZIP"
    # SHA-1 is the checksum published in Google's repository2-1.xml.
    (cd "$BASE/downloads" && printf '%s  %s\n' \
      'd313adb7aedccf6cf0cfca51ec180f0059f5f8f8' "$ZIP" | sha1sum -c -)
    STAGE="$(mktemp -d "$BASE/tools-stage.XXXXXX")"
    unzip -q "$BASE/downloads/$ZIP" -d "$STAGE"
    [[ ! -e "$ANDROID_HOME/cmdline-tools/12.0" ]]
    mv "$STAGE/cmdline-tools" "$ANDROID_HOME/cmdline-tools/12.0"
    rmdir "$STAGE"
fi
PHASE=sdk_packages
if [[ ! -f "$BASE/.sdk-installed" ]]; then
    # Accept licenses for the requested SDK packages. Preserve sdkmanager's exit
    # status rather than yes's normal SIGPIPE when sdkmanager closes its input.
    set +o pipefail
    yes | timeout --kill-after=30s 35m "$SDKMANAGER" --sdk_root="$ANDROID_HOME" --channel=0 \
      'platform-tools' 'emulator' 'system-images;android-33;google_apis;x86_64'
    set -o pipefail
    [[ -x "$ANDROID_HOME/emulator/emulator" && -x "$ANDROID_HOME/platform-tools/adb" ]]
    [[ -f "$ANDROID_HOME/system-images/android-33/google_apis/x86_64/system.img" ]]
    touch "$BASE/.sdk-installed"
else
    echo 'Reusing completed SDK installation.'
fi
PHASE=avd_preflight
# Print only relevant paths and package metadata, not the shared account's env.
printf 'ANDROID_USER_HOME=%s\nANDROID_AVD_HOME=%s\nTMPDIR=%s\n' \
  "$ANDROID_USER_HOME" "$ANDROID_AVD_HOME" "$TMPDIR"
"$JAVA_HOME/bin/java" -version
for package in emulator platform-tools system-images/android-33/google_apis/x86_64; do
    echo "SDK package: $package"
    cat "$ANDROID_HOME/$package/source.properties"
done
IMAGE_DIR="$ANDROID_HOME/system-images/android-33/google_apis/x86_64"
[[ -r "$IMAGE_DIR/system.img" ]]
if [[ ! -r "$IMAGE_DIR/userdata.img" && ! -d "$IMAGE_DIR/data" ]]; then
    echo "Incomplete system image: neither readable userdata.img nor data/ in $IMAGE_DIR" >&2
    false
fi
/usr/bin/python3 -I -B - <<'PY'
import os
import tempfile
for key in ('ANDROID_USER_HOME', 'ANDROID_AVD_HOME', 'TMPDIR'):
    directory = os.environ[key]
    with tempfile.TemporaryFile(dir=directory) as handle:
        handle.write(b'AVD write check\n')
        handle.flush()
    print('Directory write check passed:', directory, flush=True)
PY
# Only create this job's fresh AVD; never reset or edit another experiment's AVD.
PHASE=avd_create
# avdmanager hides creation warnings unless --verbose is set. Retain them in
# both the Slurm output and a dedicated log; pipefail preserves failure status.
timeout --kill-after=10s 120 "$ANDROID_HOME/cmdline-tools/12.0/bin/avdmanager" --verbose create avd \
  --name "SoftProbe${SLURM_JOB_ID}" --package 'system-images;android-33;google_apis;x86_64' \
  --device pixel_6 --path "$ANDROID_AVD_HOME/SoftProbe${SLURM_JOB_ID}.avd" \
  <<< 'no' 2>&1 | tee "$JOB_DIR/avd-create.log"
[[ -s "$ANDROID_AVD_HOME/SoftProbe${SLURM_JOB_ID}.avd/config.ini" ]]
echo 'AVD_CREATE_OK'
PHASE=kvm_report
/usr/bin/python3 -I -B "$REPO/scripts/orion/check_kvm.py" \
  --output "$ROOT/logs/android-soft-kvm-${SLURM_JOB_ID}.json"
PHASE=emulator_probe
timeout --signal=TERM --kill-after=30s 30m /usr/bin/python3 -I -B \
  "$REPO/scripts/orion/android_soft_probe.py" --sdk "$ANDROID_HOME" \
  --avd "SoftProbe${SLURM_JOB_ID}" --output "$JOB_DIR" --boot-seconds 1200
printf '{"status":"passed","stage":"complete"}\n' > "$JOB_DIR/setup-status.json"
