function generarCamposGrafico() {
    const n = parseInt(document.getElementById('n').value);
    const m = parseInt(document.getElementById('m').value);
    let html = '<h3>Función objetivo</h3><div class="fila-funcion">';

    for (let i = 1; i <= n; i++) {
        html += `<span class="grupo-variables">
                    <input class="input-campos" name="obj${i}" type="number" step="any" required>
                    <span>x<sub>${i}</sub></span>
                 </span>`;
    }

    html += '</div><h3>Restricciones</h3>';

    for (let j = 1; j <= m; j++) {
        html += '<div class="fila-restriccion">';
        for (let i = 1; i <= n; i++) {
            html += `<span class="grupo-variables">
                        <input class="input-campos" name="cons${j}_${i}" type="number" step="any" required>
                        <span>x<sub>${i}</sub></span>
                     </span>`;
        }
        html += `<select class="input-campos" name="type${j}">
                    <option value="<=">≤</option>
                    <option value=">=">≥</option>
                    <option value="=">=</option>
                 </select>
                 <input class="input-campos" name="rhs${j}" type="number" step="any" required>
                 </div>`;
    }

    document.querySelector('.btn-resolver').classList.remove('contenedor-oculto');
    document.querySelector('.btn-limpiar').classList.remove('contenedor-oculto');
    document.getElementById('campos-grafico').innerHTML = html;
}

function limpiarFormularioGrafico() {
    document.querySelector('form').reset();
    document.getElementById('campos-grafico').innerHTML = '';
    document.querySelector('.btn-resolver').classList.add('contenedor-oculto');
    document.querySelector('.btn-limpiar').classList.add('contenedor-oculto');
    const res = document.getElementById('resultado-wrapper');
    if (res) res.innerHTML = '';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Function to render the graphical solution using Plotly
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
  autosize: true,
  width: null, // Deja que Plotly use el ancho del div
  height: 600,
  margin: {l: 40, r: 40, t: 60, b: 40},
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
      width: 600,
      scale: 2
    }
  };

  Plotly.newPlot('grafico', [trace_region, trace_optimo], layout, config);

  window.addEventListener('resize', () => {
    Plotly.Plots.resize('grafico');
  });

  // Solo agrega listeners si los botones existen
  // Solo PNG, y evita descargas múltiples simultáneas
  const btnPng = document.getElementById('download-png');
  if (btnPng) {
    btnPng.addEventListener('click', async () => {
      btnPng.disabled = true;
      try {
        await Plotly.downloadImage('grafico', {format: 'png', filename: 'grafico_optimo'});
      } finally {
        btnPng.disabled = false;
      }
    });
  }
}