
function alertme(msg)
{
    alert(msg);
}

function send()
{

   var textInput = document.getElementById('text');
   var btn = document.getElementById('btn');
   var str = textInput.value;
   $.post("", { data: str });
  textInput.value="";

   alertme('data sent');

}

