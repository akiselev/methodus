from pathlib import Path
p=Path('crates/scientific/src/lib.rs')
s=p.read_text()
s=s.replace(
'''pub struct IterationTrace {
    pub iteration: usize,
    pub residual_norm: f64,
    pub scaled_residual_norm: f64,
    pub damping: f64,
}''',
'''pub struct IterationTrace {
    pub iteration: usize,
    pub residual_norm: f64,
    pub scaled_residual_norm: f64,
    pub block_scaled_residual_norms: Vec<(String, f64)>,
    pub damping: f64,
}''')
s=s.replace(
'''            trace.push(IterationTrace {
                iteration,
                residual_norm: raw,
                scaled_residual_norm: scaled,
                damping: 0.0,
            });''',
'''            trace.push(IterationTrace {
                iteration,
                residual_norm: raw,
                scaled_residual_norm: scaled,
                block_scaled_residual_norms: block_scaled_norms(layout, &residual),
                damping: 0.0,
            });''')
s=s.replace(
'''        trace.push(IterationTrace {
            iteration,
            residual_norm: raw,
            scaled_residual_norm: scaled,
            damping,
        });''',
'''        trace.push(IterationTrace {
            iteration,
            residual_norm: raw,
            scaled_residual_norm: scaled,
            block_scaled_residual_norms: block_scaled_norms(layout, &residual),
            damping,
        });''')
needle='''fn scaled_norm(layout: &BlockLayout, residual: &[f64]) -> f64 {
'''
helper='''fn block_scaled_norms(layout: &BlockLayout, residual: &[f64]) -> Vec<(String, f64)> {
    layout
        .blocks
        .iter()
        .map(|block| {
            let norm = residual[block.offset..block.offset + block.len]
                .iter()
                .map(|value| (value / block.scale).powi(2))
                .sum::<f64>()
                .sqrt();
            (block.name.clone(), norm)
        })
        .collect()
}

'''
if helper not in s:
    s=s.replace(needle,helper+needle,1)
p.write_text(s)
