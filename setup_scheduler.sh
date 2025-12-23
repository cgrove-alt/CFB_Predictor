#!/bin/bash
#
# Setup script for Sharp Sports Predictor automatic data refresh
#
# This creates a launchd job that runs the scheduler at:
# - 6 AM daily
# - 10 AM and 6 PM on Saturdays (game days)
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLIST_NAME="com.sharp-sports.refresh.plist"
PLIST_PATH="$HOME/Library/LaunchAgents/$PLIST_NAME"

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

# Create the plist file
cat > "$PLIST_PATH" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.sharp-sports.refresh</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>$SCRIPT_DIR/scheduler.py</string>
    </array>
    <key>StartCalendarInterval</key>
    <array>
        <!-- Run at 6 AM daily -->
        <dict>
            <key>Hour</key>
            <integer>6</integer>
            <key>Minute</key>
            <integer>0</integer>
        </dict>
        <!-- Run at 10 AM on Saturdays (pre-game) -->
        <dict>
            <key>Weekday</key>
            <integer>6</integer>
            <key>Hour</key>
            <integer>10</integer>
            <key>Minute</key>
            <integer>0</integer>
        </dict>
        <!-- Run at 2 PM on Saturdays (mid-day) -->
        <dict>
            <key>Weekday</key>
            <integer>6</integer>
            <key>Hour</key>
            <integer>14</integer>
            <key>Minute</key>
            <integer>0</integer>
        </dict>
        <!-- Run at 6 PM on Saturdays (evening) -->
        <dict>
            <key>Weekday</key>
            <integer>6</integer>
            <key>Hour</key>
            <integer>18</integer>
            <key>Minute</key>
            <integer>0</integer>
        </dict>
    </array>
    <key>WorkingDirectory</key>
    <string>$SCRIPT_DIR</string>
    <key>StandardOutPath</key>
    <string>$SCRIPT_DIR/logs/launchd_stdout.log</string>
    <key>StandardErrorPath</key>
    <string>$SCRIPT_DIR/logs/launchd_stderr.log</string>
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
EOF

echo "Created plist at: $PLIST_PATH"

# Unload if already loaded
launchctl unload "$PLIST_PATH" 2>/dev/null

# Load the new plist
launchctl load "$PLIST_PATH"

if [ $? -eq 0 ]; then
    echo "Scheduler loaded successfully!"
    echo ""
    echo "Schedule:"
    echo "  - Daily at 6:00 AM"
    echo "  - Saturdays at 10:00 AM, 2:00 PM, 6:00 PM"
    echo ""
    echo "To check status:  launchctl list | grep sharp-sports"
    echo "To unload:        launchctl unload $PLIST_PATH"
    echo "Logs:             $SCRIPT_DIR/logs/"
else
    echo "Failed to load scheduler. Check the plist syntax."
    exit 1
fi
