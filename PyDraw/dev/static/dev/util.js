
function alertme(msg)
{
    console.log('>>JS merge');
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

console.log("hue");
$( document ).ready(function() {
    $( "#submit_btn" ).click(function() {
//  alert( "Handler for .click() called." );
console.log('Click');
$.ajax({url: '/draw/test2',
        data:{ info :$( "#input_text" ).val()},
        success: function(result){

        console.log(result);

        $("#divul_meu").append(result.info);

    }});
});
});



function test(){
    console.log("Click2")
}