include("src/nmf_logic.jl")

face_image = load("data/face01.jpg") 
shrinked_image = imresize(face_image, (100, 100))
V = Float64.(Gray.(shrinked_image))  # transform to grayscale and convert to float matrix

for r in [10, 20, 50, 100, 200]
    println("Using $r features for approximation...")
    W, H = nmf_optimize(V, r, iterations=50)
    reconstructed_face = W * H
    error = norm(V - reconstructed_face)
    println("The error is: ", error)
end