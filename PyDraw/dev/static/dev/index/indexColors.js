var colors = ['rgb(20,20,20)', 'white', 'red', 'orange', 'green' ,'blue', 'indigo'];

//this is going to return an array with the class name swatch
var swatches = document.getElementsByClassName('swatch');

//we are caching the length of swatches because is
//less expensive to do a comparison with a variable than
//calling a method


for (var i=0, n=colors.length;i<n;i++){
    var sw = document.createElement('div');
    sw.className = 'swatch';
    sw.style.backgroundColor = colors[i];
    sw.addEventListener('click', setSwatch);

    var node = document.getElementById('colors');
    node.appendChild(sw);
}

// the classname of the selected swatch is dynamically
//modified by the following methods

function setSwatch(e)
{
    //identify swatch
    //set color
    //give active class

    var swatch = e.target; // returns the element on which the event was fired
    setHue(swatch.style.backgroundColor);
    swatch.className = 'swatch active'

}

function initColors(){
    setSwatch({target:document.getElementsByClassName('swatch')[0]});
}

function myalert(msg)
{
    alert(msg);
}

function setHue(color)
{
    console.log(color);
    context.fillStyle = color;
    context.strokeStyle = color;
    var active = document.getElementsByClassName('swatch active')[0];

    if(active){
        active.className='swatch';
    }
}