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
