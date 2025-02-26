# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import tutel_custom_kernel

if 'OP_LOADER' not in os.environ:
    os.environ['OP_LOADER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.')
