var minRad = 5,
maxRad = 15,
defaultRad = 7
interval = 1;

radSpan = document.getElementById('radval');
decRad = document.getElementById('decreaseRad');
incRad = document.getElementById('increaseRad');

function setRadius(newRadius)
{
    if(newRadius < minRad){
        newRadius = minRad;
    }
    else if(newRadius > maxRad){
        newRadius = maxRad; 
    }

    radius = newRadius;
    context.lineWidth = radius*2;

    radSpan.innerHTML = radius;
}

decRad.addEventListener('click', function(){
    setRadius(radius-interval);
})
incRad.addEventListener('click', function(){
    setRadius(radius+interval);
})