function renderGraficoSolver(vertices, optimo) {
  if (!Array.isArray(vertices) || !optimo || typeof optimo.x !== "number" || typeof optimo.y !== "number") return;

  const x = vertices.map(v => v.x);
  const y = vertices.map(v => v.y);

  const trace_region = {
    x: x.concat([x[0]]),
    y: y.concat([y[0]]),
    fill: "toself",
    type: "scatter",
    name: "Región Factible",
    hoverinfo: "x+y",
    fillcolor: "rgba(0,100,80,0.2)",
    line: {color: "green"}
  };

  const trace_optimo = {
    x: [optimo.x],
    y: [optimo.y],
    mode: "markers+text",
    marker: {color: 'red', size: 12},
    text: [`Óptimo (${optimo.x.toFixed(2)}, ${optimo.y.toFixed(2)})`],
    textposition: "top center",
    name: "Solución Óptima"
  };

  const layout = {
    title: "Solución del Método Gráfico",
    xaxis: {title: "X₁"},
    yaxis: {title: "X₂"},
    hovermode: "closest"
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    toImageButtonOptions: {
      format: 'png',
      filename: 'grafico_optimo',
      height: 600,
      width: 800,
      scale: 2
    }
  };

  Plotly.newPlot('grafico', [trace_region, trace_optimo], layout, config);

  document.getElementById('download-png').addEventListener('click', () => {
    Plotly.downloadImage('grafico', {format: 'png', filename: 'grafico_optimo'});
  });

  document.getElementById('download-pdf').addEventListener('click', () => {
    Plotly.downloadImage('grafico', {format: 'pdf', filename: 'grafico_optimo'});
  });
}