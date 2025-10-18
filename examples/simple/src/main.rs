use itertools::Itertools;
use luminal::prelude::*;
//use luminal_2::{
// codegen::codegen, extract::search, run::run_graph, translate::translate_graph,
// utils::build_search_space, GPUArch,
//};
use luminal_nn::Linear;
use luminal_nn::Conv1D;
use rand::{rng, Rng};

fn main() {
    let mut rng = rng();
    // let weight = (0..64 * 1).map(|_| rng.random()).collect_vec();
    // Create a new graph
    let mut cx = Graph::new();
    // Randomly initialize a linear layer with an input size of 4 and an output size of 5
    // let model = Linear::new(4, 20, false, &mut cx);
    let model = Conv1D::new(8, 4, 2, 2, 1, 0, false, &mut cx);
    // model.weight.set(weight.clone());
    model.weight.set(vec![
        -0.1700, -0.2000, 0.1000, -0.0200, 0.1000, 0.0200, -0.2100, -0.2300, -0.0600, 0.1500,
        0.1200, 0.1000, 0.1800, 0.0600, -0.1700, -0.0400, 0.1000, -0.0200, -0.1700, 0.1000,
        0.1100, 0.1600, 0.2000, 0.0100, -0.0500, 0.2100, -0.0200, 0.0300, -0.0900, -0.0500,
        0.1600, 0.0400, 0.0400, -0.1700, 0.1100, 0.0600, -0.1200, -0.2300, 0.2300, -0.2100,
        -0.2200, 0.1100, -0.0100, -0.1400, 0.1700, 0.0300, 0.1000, -0.1400, -0.2100, -0.1800,
        0.2000, -0.2300, -0.1600, 0.2200, 0.0900, 0.0700, -0.1000, -0.0400, -0.0500, 0.1400,
        0.0700, -0.1200, 0.1400, 0.2200,
    ]);
    // Make an input tensor
    let a = cx.tensor((8, 12));
    a.set(vec![
        1., 2., 6., 4., 8., 1., 6., 0., 1., 0., 6., 4., 3., 4., 9., 3., 8., 8., 5., 5., 0., 4.,
        2., 7., 6., 4., 2., 2., 8., 0., 7., 3., 0., 0., 7., 2., 3., 3., 1., 9., 5., 4., 5., 5.,
        8., 0., 0., 1., 2., 1., 8., 9., 4., 7., 7., 6., 8., 5., 0., 9., 1., 6., 0., 1., 4., 3.,
        3., 5., 8., 7., 9., 5., 6., 5., 6., 9., 7., 0., 9., 5., 6., 0., 6., 1., 2., 1., 0., 1.,
        3., 6., 8., 0., 6., 6., 3., 2.,
    ]);
    // .set(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]);
    // Feed tensor through model
    let b = model.forward(a).retrieve();

    // Execute the graph
    cx.execute_debug();
    println!("B: {:?}", b.data());
}
