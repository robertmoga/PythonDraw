var sendBtn = document.getElementById('sendBtn');

sendBtn.addEventListener('click', sendData);

function sendData()
{
    var imgData = canvas.toDataURL();
    $.post("", { data: imgData });
//    window.open(imgData, '_blank', 'location=0, menubar=0');
    alert("Data sent  " );

}