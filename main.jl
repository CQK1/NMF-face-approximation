include("src/nmf_logic.jl")
include("src/visualization.jl")

face_image = load("data/face01.jpg") 
shrinked_image = imresize(face_image, (100, 100))
V = Float64.(Gray.(shrinked_image))  # transform to grayscale and convert to float matrix

for r in [200]
    println("Using $r features for approximation...")
    W, H = NMFLogic.nmf_optimize(V, r, iterations=50)
    reconstructed_face = W * H
    error = norm(V - reconstructed_face)
    println("The error is: ", error)
    Visualization.visualize_reconstruction(V, reconstructed_face)
end