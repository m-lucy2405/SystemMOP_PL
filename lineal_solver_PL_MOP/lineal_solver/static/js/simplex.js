/* ========= genera inputs de la FO y restricciones ========= */
function generarCampos() {
    const n = +document.getElementById('n').value,
          m = +document.getElementById('m').value;

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
    document.getElementById('campos').innerHTML = html;
}

/* ========= limpia todo el formulario y la sección de resultados ========= */
function limpiarFormulario() {
    document.querySelector('form').reset();
    document.getElementById('campos').innerHTML = '';
    document.querySelector('.btn-resolver').classList.add('contenedor-oculto');
    document.querySelector('.btn-limpiar').classList.add('contenedor-oculto');
    const res = document.getElementById('resultado-wrapper');
    if (res) res.remove();

    window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* ========= MathJax ========= */
window.MathJax = {
    tex: {
        inlineMath: [['\\(', '\\)'], ['\\[', '\\]']]
    },
    svg: { fontCache: 'global' }
};

/* ========= resaltado de pivote (opcional si usas pivot-cell) ========= */
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.pivot-cell').forEach(c => {
        c.style.background = '#fffacd';
        c.style.fontWeight = 'bold';
    });
});

/* ========= evita reenvío al recargar ========= */
if (window.history.replaceState) {
    window.history.replaceState(null, '', window.location.pathname);
}