#!/usr/bin/env bash
read -p "Delete existing checkpoints and log files (y/n)?" choice
case "$choice" in
  y|Y )
	echo "Deleting existing checkpoints and log files..."
	rm checkpoints/*.ckpt
	rm -rf logging/tuna-log/
	echo "Done!"
	;;
  n|N )
	echo "no"
	;;
  * )
	echo "invalid"
	;;
esac


