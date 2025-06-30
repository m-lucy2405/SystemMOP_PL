/* Google Translate init */
function googleTranslateElementInit() {
    new google.translate.TranslateElement({
        pageLanguage: 'es',
        autoDisplay: false
    }, 'google_translate_element');
}

/* Script para disparar traducciones desde tus botones */
function translatePage(lang) {
    var combo = document.querySelector('#google_translate_element select');
    if (!combo) return;
    combo.value = lang;
    combo.dispatchEvent(new Event('change'));
}
document.getElementById('btn-translate-es')
    .addEventListener('click', function () { translatePage('es'); });
document.getElementById('btn-translate-en')
    .addEventListener('click', function () { translatePage('en'); });
document.getElementById('btn-translate-fr')
    .addEventListener('click', function () { translatePage('fr'); });