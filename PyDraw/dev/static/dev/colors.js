var colors = ['black', 'white', 'red', 'orange', 'green' ,'blue', 'indigo'];


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

    var swatch = e.target // returns the element on which the event was fired 
    setColor(swatch.style.backgroundColor);

    swatch.className += ' active'
    //alert('am iesit din set swatch');
}

function setColor(color)
{
    //alert('am intrat in set color');
    context.fillStyle = color;
    context.strokeStyle = color;
    var active = document.getElementsByClassName('active')[0];

    if(active){
        active.className='swatch';
    }
    //alert('am iesit din set color');
}

function initColors(){
    setSwatch({target:document.getElementsByClassName('swatch')[0]});
}
