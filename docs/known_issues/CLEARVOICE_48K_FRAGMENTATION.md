# ClearVoice MossFormer2_SE_48K VRAM Fragmentation Issue

**Status:** Deferred to v1.8
**Severity:** High (causes Pass 1 failure on 8GB GPUs)
**Affected Component:** `clearvoice:MossFormer2_SE_48K` speech enhancer
**Workaround:** Use `zipenhancer:torch` instead (recommended)

## Summary

The ClearVoice MossFormer2_SE_48K model causes CUDA memory fragmentation on GPUs with 8GB VRAM (e.g., RTX 3060). After processing 7-8 scenes, PyTorch's caching allocator becomes too fragmented to allocate contiguous memory blocks, causing OOM errors even when total VRAM usage appears low in Task Manager.

## Symptoms

1. Enhancement succeeds for scenes 1-7
2. Scene 8+ fails with "CUDA error: out of memory"
3. Task Manager shows VRAM is NOT near capacity
4. CUDA context becomes corrupted
5. Subsequent ASR model loading fails
6. Pass 1 fails, Pass 2 succeeds (fresh subprocess)

## Root Cause Analysis

### Why Task Manager Shows Nothing Wrong

- Windows Task Manager shows **total reserved VRAM** (virtual address space)
- PyTorch's internal free-list is **fragmented** (invisible to Windows)
- OOM occurs when no **contiguous** block is available for the next allocation
- `torch.cuda.empty_cache()` only returns unused blocks, doesn't defragment

### Why Scene 8?

- MossFormer2_SE_48K processes audio at 48kHz (3x more data than 16kHz models)
- Each scene creates/destroys large intermediate tensors
- Memory fragmentation accumulates across scenes
- By scene 8, fragmentation exceeds threshold for large allocations
- Longer scenes (up to 221.9s in test) trigger the issue earlier

### Technical Details

```
Scene 1: Model loads, allocates ~3-4GB, processes, frees
Scene 2-7: Reuses model, allocates/frees tensors, fragmentation grows
Scene 8: Allocation request fails - no contiguous block available
         Despite total VRAM < 8GB, free-list is Swiss cheese
```

## Mitigation (Current v1.7.x)

1. **GUI:** MossFormer2_SE_48K disabled in dropdown with tooltip
2. **Default:** ZipEnhancer marked as recommended
3. **Test Suite:** Uses ffmpeg-dsp instead of clearvoice for Test 2

## Potential Solutions for v1.8

### Option A: Subprocess Isolation (Recommended)
Run each scene's enhancement in a separate subprocess. Fresh CUDA allocator per scene prevents fragmentation accumulation.

**Pros:** Guaranteed fix, clean architecture
**Cons:** ~2s overhead per scene (model loading)

### Option B: PyTorch Configuration
Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to allow PyTorch to request new memory segments when fragmented.

**Pros:** No code changes, might work
**Cons:** Untested, may not fully solve the issue

### Option C: Model Reload Per Scene
Destroy and recreate ClearVoice model after every N scenes to reset allocator state.

**Pros:** Simpler than subprocess
**Cons:** Same overhead as Option A, doesn't guarantee fix

### Option D: Batch by Duration
Group short scenes together, process long scenes alone with model reload between batches.

**Pros:** Reduces overhead
**Cons:** Complex logic, unpredictable results

## Diagnostic Tests for v1.8

### Test 1: Log VRAM Stats
```python
# Before/after each scene:
allocated = torch.cuda.memory_allocated() / 1024**2
reserved = torch.cuda.memory_reserved() / 1024**2
logger.debug(f"Scene {n}: allocated={allocated:.1f}MB, reserved={reserved:.1f}MB")
```

If `reserved` grows but `allocated` stays low = fragmentation confirmed.

### Test 2: Expandable Segments
```python
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

Run full test suite. If all scenes succeed, this is the simplest fix.

### Test 3: Subprocess Isolation
Implement Solution A and verify no fragmentation across 20+ scenes.

## Related Files

- `whisperjav/modules/speech_enhancement/backends/clearvoice.py`
- `whisperjav/modules/speech_enhancement/pipeline_helper.py`
- `whisperjav/pipelines/balanced_pipeline.py`
- `whisperjav/webview_gui/assets/index.html` (dropdown disabled)

## Commits

- `b54dde7` - Added try/except for CUDA cache clear failures (partial fix)
- (this commit) - Disabled MossFormer2_SE_48K in GUI

## References

- PyTorch CUDA Memory Management: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
- CUDA Caching Allocator: https://pytorch.org/docs/stable/notes/cuda.html#use-of-custom-cuda-allocators
