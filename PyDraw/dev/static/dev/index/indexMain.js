var canvas = document.getElementById('canvas');
canvas.style.background = 'beige';
var context = canvas.getContext('2d');

canvas.width = innerWidth;
canvas.height = 600;
var dragging=false;
var radius = 6;
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
    destroy_elements();
}

function destroy_elements()
    {
        var myNode = document.getElementById("content");
        while (myNode.firstChild) {
        myNode.removeChild(myNode.firstChild);
        }
    }