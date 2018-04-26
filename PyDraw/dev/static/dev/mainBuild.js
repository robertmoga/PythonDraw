var canvas = document.getElementById('canvas');
canvas.style.background = 'black';
var context = canvas.getContext('2d');


canvas_dim = 200;


canvas.width = canvas_dim;
canvas.height = canvas_dim;
var dragging=false;
var radius = 7;
context.lineWidth = radius*2;
/*
every time when is called [designed to be called
when the process is engaged] .
Every time we put a point, we start a path
from the actual position of the mouse
and draw a line to the next position of the
mouse when putPoint is called again
*/

function setColor(color){
context.fillStyle = color
context.strokeStyle = color
}

function putPoint(e) {
    if (dragging) {
        context.lineTo(e.offsetX,e.offsetY);
        context.stroke();
        context.beginPath();
        x = e.offsetX;
        y = e.offsetY;
        startAngle = 0;
        endAngle = Math.PI * 2;
        context.arc(x, y, radius, startAngle, endAngle);
        context.fill();

        context.beginPath();
        context.moveTo(e.offsetX,e.offsetY);

    }
}

/*
following methods are engaging and disengaging
the process of dragging lines
*/
function engage(e) {
    dragging = true;
    putPoint(e);
}

function disengage() {
    dragging = false;
    context.beginPath();
}

canvas.addEventListener('mousedown', engage);
canvas.addEventListener('mouseup', disengage);
canvas.addEventListener('mousemove', putPoint);
canvas.addEventListener('mouseleave',disengage);

var clearBtn = document.getElementById('clearBtn');
clearBtn.addEventListener('click',clear);

function myAlert(msg) {
    alert(msg);
}

function clear()
{
    context.clearRect(0,0,canvas.width, canvas.height);
}

var sendBtn = document.getElementById('sendBtn');
sendBtn.addEventListener('click', sendData);

function sendData()
{
      var inputLabel = document.getElementById('labelInput');
      var str_label = inputLabel.value;

    if( str_label.length == 1)
    {
        var imgData = canvas.toDataURL();
        $.post("", { imageData: imgData , imageLabel : str_label});
//        alert("Data sent  " );
    }
    else
    {
        alert('Label is not valid');
    }

}