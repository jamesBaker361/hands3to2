import pyvista as pv

# Download an example mesh (e.g., the bunny mesh)
mesh = pv.examples.download_bunny_coarse()

# Create a plotter object to manage multiple views
plotter = pv.Plotter(off_screen=True)

# Add the mesh to the plotter
plotter.add_mesh(mesh, color='lightblue')

# Set up different camera positions (elevation, azimuth, roll)
views = [
    (1, 1, 1),   # View 1: Diagonal view from the top
    (0, 0, 1),   # View 2: Top-down view
    (1, 0, 0),   # View 3: Side view
    (0, 1, 0)    # View 4: Front view
]

# Render the mesh from different perspectives
for i, view in enumerate(views):
    plotter.camera_position = view
    plotter.show(auto_close=False)  # Show but don't close the plotter to keep rendering

    # Save each perspective as an image
    plotter.screenshot(f"render_view_{i + 1}.png")

# Close the plotter once all renders are done
plotter.close()
