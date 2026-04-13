include("src/nmf_logic.jl")
include("src/visualization.jl")
using Images, FileIO, LinearAlgebra

data_dir = "data"
image_files = filter(f -> endswith(f, ".jpg"), readdir(data_dir))
println("Discovered $(length(image_files)) image(s) in the data directory: ", image_files)

r = 200

for fname in image_files
    println("Using $r features for approximation...")

    img_path = joinpath(data_dir, fname)
    face_image = FileIO.load(img_path)
    V = Float64.(Gray.(face_image))  # transform to grayscale and convert to float matrix

    W, H = NMFLogic.nmf_optimize(V, r, iterations=50)
    reconstructed_face = W * H
    error = LinearAlgebra.norm(V - reconstructed_face)
    println("The error is: ", error)

    label = splitext(fname)[1]
    Visualization.visualize_reconstruction(V, reconstructed_face, label)
end
