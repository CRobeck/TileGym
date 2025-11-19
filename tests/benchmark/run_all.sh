#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Run all Python benchmark files
cd "$(dirname "$0")"

for file in *.py; do
    echo "Running $file..."
    python "$file"
    echo "---"
done
