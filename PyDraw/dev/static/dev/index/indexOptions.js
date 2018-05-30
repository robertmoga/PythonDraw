console.log(">> Options loaded");
var OPTION = 'NULL';
var options = ['Lines and words', 'Words and chars'];


for (var i = 0, n = options.length; i < n; i++) {
    var sw = document.createElement('div');
    sw.className = 'toggle';
    var text = document.createTextNode(options[i]);
    sw.appendChild(text);
    sw.addEventListener('click', setOption);

    var node = document.getElementById('options');
    node.appendChild(sw);
}

function setOption(e) {
    var swatch = e.target;
    setActive();
    swatch.className = 'toggle active'
    console.log("Test : " + swatch.textContent + "  " + options[0]);
    if (swatch.textContent == options[0]) {
        OPTION = 'lines';
    }
    else {
        OPTION = 'words';
    }

}


function setActive() {
    // console.log(color);
    var active = document.getElementsByClassName('toggle active')[0];

    if (active) {
        active.className = 'toggle';
    }
}

function initOptions() {
    console.log("Optiune setata");
    setActive({ target: document.getElementsByClassName('toggle')[0] });
}