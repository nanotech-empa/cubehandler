# Cubehandler

Tools for reducing, combining and rendering Gaussian cube files.

The rendering command accepts the NGL camera matrix, signed isovalue/colour pairs,
image dimensions, background and opacity. Run `cubehandler render --help` for the
complete interface. Headless rendering requires working EGL or OSMesa on the
execution node.

Use `--slice-plane plane.json` to include a visible, coloured plane in the 3D
render. Example JSON (coordinates and lengths in angstrom):

```json
{
  "center": [0.0, 0.0, 0.65],
  "normal": [0.0, 0.0, 1.0],
  "offset": 0.0,
  "width": 8.0,
  "height": 8.0,
  "resolution": 64,
  "vmin": -0.03,
  "vmax": 0.03,
  "periodic": false
}
```

The plane uses the EMPA viewer's linear interpolation, origin and boundary
conventions and `coolwarm` colours. Colour limits are recorded with the saved
view. Cells outside an isolated grid are omitted; periodic sampling is explicit.
The chosen input cube supplies the values, so full-resolution rendering samples
the full cube instead of a stored preview. Omitting `--slice-plane` keeps the
previous isosurface-only behavior. AiiDA applications should capture this JSON
in their render CalcJob's inputs and retrieve the resulting image and log.

Use `--lighting lighting.json` for atom and isosurface lighting, for example
`{"ambient": 0.45, "direct": 0.55, "gloss": 0.15}`. Each value lies between 0 and
1. The plane remains unlit so its data colours are independent of illumination.
These controls map to VTK's material coefficients; NGL previews use a different
lighting model, so their shading is comparable rather than pixel-identical.
Omitting `--lighting` retains the original renderer defaults.

## Software attribution and scientific acknowledgements

Please credit [cubehandler](https://github.com/nanotech-empa/cubehandler) and its
[contributors](https://github.com/nanotech-empa/cubehandler/graphs/contributors),
and identify the release or immutable commit used. Existing author,
contributor and funding acknowledgements remain authoritative.

This revision has no top-level licence file. Preserve existing source
notices and confirm applicable reuse terms with the maintainers; this
citation guidance does not supply a licence.

Credit cubehandler and its contributors for reused cube-file operations and cite the underlying calculation software and methods as applicable. Standalone use does not imply use of AiiDA or AiiDAlab.

The [shared credit guide](https://github.com/nanotech-empa/.github/blob/main/CITATION_GUIDANCE.md) keeps software attribution, licence
compliance and scientific citation requests separate. Scientific citation
requests are not additional MIT licence conditions.
