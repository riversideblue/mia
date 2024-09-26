#!/bin/bash

echo -e " ____   ____  ___     _____           __  ___    ___   \n" \
     "|    \\ l    j|   \\   / ___/          /  ]|   \\  |   \\  \n" \
     "|  _  Y |  T |    \\ (   \\_  _____   /  / |    \\ |    \\ \n" \
     "|  |  | |  | |  D  Y \\__  T|     | /  /  |  D  Y|  D  Y \n" \
     "|  |  | |  | |     | /  \\ |l_____j/   \\_ |     ||     | \n" \
     "|  |  | j  l |     | \\    |       \\     ||     ||     | \n" \
     "l__j__j|____jl_____j  \\___j        \\____jl_____jl_____j \n"

echo "Host: $(hostname)"
echo "Kernel: $(uname -r)"
echo ""

# シェルを起動
exec "/bin/bash"