

module Visualization
    using Images, FileIO, Plots
    export visualize_reconstruction
    function visualize_reconstruction(original, reconstructed)
        reconstructed_img = colorview(Gray, clamp.(reconstructed, 0, 1))
        save("data/reconstructed_face.png", reconstructed_img)
        p1 = plot(Gray.(original), title="Original", axis=false)
        p2 = plot(reconstructed_img, title="NMF Approximation", axis=false)
        plot(p1, p2, layout=(1, 2))
        display(plot) # display the plot
        gui()
    end
end