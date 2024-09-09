#!/bin/bash

echo ""
echo "Host: $(hostname)"
echo "Kernel: $(uname -r)"
echo ""

# シェルを起動
exec "/bin/bash"