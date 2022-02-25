#!/usr/bin/env bash

d=$1
f=`ls -lt $d | tail -n +2 | head -1 | awk 'BEGIN{FS=" "}{print $NF}'`
cp $d/$f $d"/last.ckpt"
echo "Copied latest ckpt file "$d/$f " to last.ckpt"
