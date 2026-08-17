#!/usr/bin/env python3
"""Self-pad block-table row tails (layer-3 splice hazard, factor 1 of 2).

BlockTable.append_row/move_row write only [0:num_blocks] of a row; the tail
keeps the PREVIOUS tenant's block ids. Tails are harmless while every
consumer respects exact seq bounds -- but the DSA indexer's expand kernel
copies FULL row width, and any transiently stale seq_len then walks past
the boundary into another request's pages. All probe prompts share the
manifest header, so foreign pages score high in the indexer and enter the
top-2048 -> the observed cross-item splices and displaced continuations
(NOTES_longctx_copy_fidelity.md, layer 3).

Fix: after append_row writes, fill the tail with the row's LAST VALID block
id. Over-reads then land in the request's own boundary block --
content-neutral regardless of any seq_len staleness. move_row copies the
padded row wholesale afterwards; swap_row swaps full rows. Cost: one numpy
slice fill per append (us-scale, CPU).
"""

import os
import sys

import vllm

VLLM = os.path.dirname(vllm.__file__)
BT = "v1/worker/block_table.py"


def edit(path, old, new, label):
    full = os.path.join(VLLM, path)
    src = open(full).read()
    if new in src:
        print(f"  = {label} (already applied)")
        return
    if old not in src:
        print(f"  ! {label}: ANCHOR NOT FOUND", file=sys.stderr)
        sys.exit(1)
    if src.count(old) != 1:
        print(f"  ! {label}: anchor not unique", file=sys.stderr)
        sys.exit(1)
    open(full, "w").write(src.replace(old, new))
    print(f"  + {label}")


edit(
    BT,
    "        num_blocks = len(block_ids)\n"
    "        start = self.num_blocks_per_row[row_idx]\n"
    "        self.num_blocks_per_row[row_idx] += num_blocks\n"
    "        self.block_table.np[row_idx, start : start + num_blocks] = block_ids",
    "        num_blocks = len(block_ids)\n"
    "        start = self.num_blocks_per_row[row_idx]\n"
    "        self.num_blocks_per_row[row_idx] += num_blocks\n"
    "        self.block_table.np[row_idx, start : start + num_blocks] = block_ids\n"
    "        # Self-pad the tail with the last valid block id: stale ids from\n"
    "        # a previous tenant otherwise survive here, and a consumer that\n"
    "        # over-reads (full-width row copies + transiently stale seq_len)\n"
    "        # would walk into another request's KV pages. Padding with our\n"
    "        # own boundary block makes over-reads content-neutral. See\n"
    "        # patch_bt_selfpad.py.\n"
    "        end = start + num_blocks\n"
    "        if num_blocks > 0 and end < self.block_table.np.shape[1]:\n"
    "            self.block_table.np[row_idx, end:] = self.block_table.np[\n"
    "                row_idx, end - 1\n"
    "            ]",
    "block_table: self-pad row tails",
)

edit(
    BT,
    "    def move_row(self, src: int, tgt: int) -> None:\n"
    "        num_blocks = self.num_blocks_per_row[src]\n"
    "        block_table_np = self.block_table.np\n"
    "        block_table_np[tgt, :num_blocks] = block_table_np[src, :num_blocks]\n"
    "        self.num_blocks_per_row[tgt] = num_blocks",
    "    def move_row(self, src: int, tgt: int) -> None:\n"
    "        num_blocks = self.num_blocks_per_row[src]\n"
    "        block_table_np = self.block_table.np\n"
    "        # Full-row copy so the self-padded tail moves along (the sliced\n"
    "        # copy left the target's previous tenant's tail in place).\n"
    "        block_table_np[tgt] = block_table_np[src]\n"
    "        self.num_blocks_per_row[tgt] = num_blocks",
    "block_table: move full padded rows",
)

print("Applied. No env gate: strictly-safer content, negligible cost.")
