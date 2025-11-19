#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Helper script to build TileGym Transformers Docker container from any location

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Find tilegym repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TILEGYM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo -e "${GREEN}TileGym Transformers Docker Build${NC}"
echo "================================"
echo ""

# Verify we're in the right place
if [ ! -f "${TILEGYM_ROOT}/setup.py" ]; then
    echo -e "${RED}ERROR: Cannot find setup.py at ${TILEGYM_ROOT}${NC}"
    echo "Please ensure this script is in modeling/transformers/ directory"
    exit 1
fi

echo -e "${GREEN}✓${NC} Found TileGym repository at: ${TILEGYM_ROOT}"
echo ""

# Build the container
echo -e "${YELLOW}Building Docker container...${NC}"
cd "${TILEGYM_ROOT}"
docker build -t tilegym-transformers -f modeling/transformers/Dockerfile .

echo ""
echo -e "${GREEN}✓ Build successful!${NC}"
echo ""
echo "Enter interactive mode:"
echo "  docker run --gpus all -it tilegym-transformers bash"
echo ""
echo "Or run inference directly:"
echo "  docker run --gpus all -it tilegym-transformers \\"
echo "    python infer.py --model_id deepseek-ai/DeepSeek-V2-Lite-Chat --use_tilegym --use_cutile --show_outputs"
