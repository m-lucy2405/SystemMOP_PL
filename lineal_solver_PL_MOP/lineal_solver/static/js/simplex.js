/* ========= genera inputs de la FO y restricciones ========= */
function generarCampos() {
    const n = +document.getElementById('n').value,
        m = +document.getElementById('m').value;
    let html = '<h3>Función objetivo</h3>';
    for (let i = 1; i <= n; i++)
        html += `<input class="input-campos" name="obj${i}" type="number" step="any" required> x<sub>${i}</sub> `;
    html += '<h3>Restricciones</h3>';
    for (let j = 1; j <= m; j++) {
        for (let i = 1; i <= n; i++)
            html += `<input class="input-campos" name="cons${j}_${i}" type="number" step="any" required> x<sub>${i}</sub> `;
        html += `<select class="input-campos" name="type${j}">
                            <option value="<=">≤</option>
                            <option value=">=">≥</option>
                            <option value="=">=</option>
                         </select>
                         <input class="input-campos" name="rhs${j}" type="number" step="any" required><br>`;
    }
    document.getElementById('campos').innerHTML = html;
}

/* ========= limpia todo el formulario y la sección de resultados ========= */
function limpiarFormulario() {
    document.querySelector('form').reset();
    document.getElementById('campos').innerHTML = '';
    const res = document.getElementById('resultado-wrapper');
    if (res) res.remove();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* MathJax */
window.MathJax = { tex: { inlineMath: [['\\(', '\\)'], ['\\[', '\\]']] }, svg: { fontCache: 'global' } };

/* 1️⃣ resaltado manual de pivote – opcional si agregas <td class="pivot-cell"> en backend */
    document.addEventListener('DOMContentLoaded', () => {
      document.querySelectorAll('.pivot-cell').forEach(c => {
        c.style.background = '#fffacd';
        c.style.fontWeight = 'bold';
      });
    });

    /* 2️⃣ evita diálogo de reenvío al recargar */
    if (window.history.replaceState) {
      window.history.replaceState(null, '', window.location.pathname);
    }