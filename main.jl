include("src/nmf_logic.jl")
include("src/visualization.jl")
using Images, FileIO, LinearAlgebra

face_image = FileIO.load("data/face01.jpg") 
V = Float64.(Gray.(face_image))  # transform to grayscale and convert to float matrix

for r in [200]
    println("Using $r features for approximation...")
    W, H = NMFLogic.nmf_optimize(V, r, iterations=50)
    reconstructed_face = W * H
    error = LinearAlgebra.norm(V - reconstructed_face)
    println("The error is: ", error)
    Visualization.visualize_reconstruction(V, reconstructed_face)
end