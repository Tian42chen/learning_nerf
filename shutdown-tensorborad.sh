#!/bin/bash
# 获取监听在 localhost:x11-6 的进程的 PID
pid=$(lsof -i:6006 | grep LISTEN | awk '{print $2}')

# 如果找到了进程
if [ -n "$pid" ]; then
    echo "Killing process with PID: $pid"
    kill $pid
else
    echo "No process found listening on localhost:x11-6"
fi
