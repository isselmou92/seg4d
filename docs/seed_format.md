# `seed_boxes.json` schema

This file tells the dynamic step where each organ is on **frame 0** of every
2D dynamic sequence. The video tracker uses these prompts as the only
manual input and propagates the masks through all remaining frames
automatically.

The file is a JSON object keyed by sequence name. Every sequence holds
three organ entries: `liver`, `kidney_right`, `kidney_left`. Each organ
entry is one of:

1. `null` -- no seed for this organ in this sequence (i.e. organ not
   visible).
2. **legacy box-only** -- a 4-element list `[x1, y1, x2, y2]` (image-space
   pixels). `[0, 0, 0, 0]` is treated as "no seed".
3. **extended** -- a dict with any of:
   - `"box"`: `[x1, y1, x2, y2]`
   - `"positive"`: `[[x, y], [x, y], ...]` (foreground points)
   - `"negative"`: `[[x, y], [x, y], ...]` (background points)

## Coordinate system

`(x, y)` are pixel coordinates of the JPEG export of frame 0. `x` runs
left-to-right, `y` runs top-to-bottom; both are zero-indexed. The
`seg4d generate-seeds` subcommand draws a 50-px grid on top of frame 0 so
you can read coordinates directly off the reference image.

## Backend support

| Backend  | `box` | `positive` | `negative` |
| -------- | :---: | :--------: | :--------: |
| MedSAM2  | yes   | yes        | yes        |
| SAM3     | -     | yes        | yes        |

If MedSAM2 receives points only, the box can be omitted. Conversely, points
with no box are treated as point-only seeds.

## Examples

### Box-only (MedSAM2)

```json
{
  "60_COR_b": {
    "liver":        [120, 80, 320, 220],
    "kidney_right": [80, 220, 180, 330],
    "kidney_left":  [340, 220, 440, 330]
  },
  "60_SAG_b": {
    "liver":        [110, 90, 330, 230],
    "kidney_right": [0, 0, 0, 0],
    "kidney_left":  [310, 230, 410, 340]
  }
}
```

### Box + points (MedSAM2)

```json
{
  "60_COR_b": {
    "liver": {
      "box":      [120, 80, 320, 220],
      "positive": [[200, 150]],
      "negative": [[60, 60]]
    },
    "kidney_right": [80, 220, 180, 330],
    "kidney_left":  null
  }
}
```

### Points only (works with both MedSAM2 and SAM3)

```json
{
  "60_COR_b": {
    "liver":        { "positive": [[200, 150]], "negative": [[60, 60]] },
    "kidney_right": { "positive": [[130, 280]] },
    "kidney_left":  { "positive": [[390, 280]] }
  }
}
```

## Auto-pairing in the interactive picker

When the SAM3 picker (`--interactive`) is run for `60_COR_b` or `60_SAG_b`,
the placed points are also copied to the matching free-breathing sequence
(`FB_COR_b` / `FB_SAG_b`) because both share the same imaging plane. You
can override the copy by re-prompting the FB sequence afterwards.
