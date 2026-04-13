

module Visualization
    using Images, FileIO, Plots
    export visualize_reconstruction
    function visualize_reconstruction(original, reconstructed, label)
        reconstructed_img = colorview(Gray, clamp.(reconstructed, 0, 1))
        save_path = "data/reconstructed_$(label).png"
        save(save_path, reconstructed_img)
        p1 = plot(Gray.(original), title="Original", axis=false)
        p2 = plot(reconstructed_img, title="NMF Approximation", axis=false)
        combined_plot = plot(p1, p2, layout=(1, 2))
        savefig(combined_plot, "data/comparison_$(label).png")
        display(combined_plot) # display the plot
        gui()
    end
end