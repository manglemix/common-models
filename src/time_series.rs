use std::fmt::Debug;

use burn::{
    config::Config,
    module::Module,
    nn::{
        gru::{Gru, GruConfig},
        Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
    },
    tensor::{backend::Backend, Tensor},
};

use crate::Model;

use super::{Activation, ThreeTuple};

#[derive(Config)]
pub struct GruNetworkConfig {
    pub grus: Vec<(GruConfig, Option<LayerNormConfig>, Activation)>,
    pub linears: Vec<(LinearConfig, Option<LayerNormConfig>, Activation)>,
    pub dropout: DropoutConfig,
}

impl GruNetworkConfig {
    pub fn new_basic(
        activation: Activation,
        d_input: usize,
        d_output: usize,
        hidden_size: usize,
        gru_count: usize,
        linear_count: usize,
        dropout_prob: f64,
        bias: bool,
        normalize: bool,
        norm_eps: f64,
    ) -> Self {
        Self {
            grus: (0..gru_count)
                .into_iter()
                .map(|i| {
                    let gru = if i == 0 {
                        GruConfig::new(d_input, hidden_size, bias)
                    } else {
                        GruConfig::new(hidden_size, hidden_size, bias)
                    };
                    (
                        gru,
                        normalize.then(|| LayerNormConfig::new(hidden_size).with_epsilon(norm_eps)),
                        activation,
                    )
                })
                .collect(),
            linears: (0..linear_count)
                .into_iter()
                .map(|i| {
                    let mut norm = None;
                    let linear = if i == linear_count - 1 {
                        LinearConfig::new(hidden_size, d_output)
                    } else {
                        norm = normalize
                            .then(|| LayerNormConfig::new(hidden_size).with_epsilon(norm_eps));
                        LinearConfig::new(hidden_size, hidden_size)
                    };
                    (linear, norm, activation)
                })
                .collect(),
            dropout: DropoutConfig::new(dropout_prob),
        }
    }
}

#[derive(Module)]
pub struct GruNetwork<B: Backend> {
    grus: Vec<ThreeTuple<Gru<B>, Option<LayerNorm<B>>, Activation>>,
    linears: Vec<ThreeTuple<Linear<B>, Option<LayerNorm<B>>, Activation>>,
    dropout: Dropout,
}

// unsafe impl<B: Backend> Send for GruNetwork<B> {}
// unsafe impl<B: Backend> Sync for GruNetwork<B> {}

impl<B: Backend> Debug for GruNetwork<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GruModule")
            .field("grus", &self.grus)
            .field("linear", &self.linears)
            .field("dropout", &self.dropout)
            .finish()
    }
}

impl<B: Backend> Model<B> for GruNetwork<B> {
    type Input = Tensor<B, 3>;
    type Output = Tensor<B, 2>;
    type Config = GruNetworkConfig;

    fn forward(&self, input: Self::Input) -> Self::Output {
        let mut x = input;
        for layer in &self.grus {
            x = layer.0.forward(x, None);
            if let Some(norm) = &layer.1 {
                x = norm.forward(x);
            }
            x = layer.2.forward(x);
            x = self.dropout.forward(x);
        }
        let [batch_size, seq_length, hidden_size] = x.dims();
        x = x.slice([0..batch_size, (seq_length - 1)..seq_length, 0..hidden_size]);
        let mut x = x.reshape([batch_size, hidden_size]);
        for layer in &self.linears {
            x = layer.0.forward(x);
            if let Some(norm) = &layer.1 {
                x = norm.forward(x);
            }
            x = layer.2.forward(x);
            x = self.dropout.forward(x);
        }
        // let [batch_size, a, b] = x.dims();
        // x.reshape([batch_size, a * b])
        x
    }

    fn from_config(config: Self::Config, device: &B::Device) -> Self {
        Self {
            grus: config
                .grus
                .into_iter()
                .map(|(a, b, c)| ThreeTuple(a.init(device), b.map(|x| x.init(device)), c))
                .collect(),
            linears: config
                .linears
                .into_iter()
                .map(|(a, b, c)| ThreeTuple(a.init(device), b.map(|x| x.init(device)), c))
                .collect(),
            dropout: config.dropout.init(),
        }
    }
}
