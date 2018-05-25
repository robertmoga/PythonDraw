$(function() {

// nothing

console.log("nothing")

});

$('#post-form').on('submit', function(event){
    event.preventDefault();
    console.log("form submitted!")  // sanity check
    create_post();
});success : function(json) {
    $('#post-text').val(''); // remove the value from the input
    console.log(json); // log the returned json to the console
    $("#talk").prepend("<li><strong>"+json.text+"</strong> - <em> "+json.author+"</em> - <span> "+json.created+"</span></li>");
    console.log("success"); // another sanity check
},

function myAlert(msg)
{
    alert(msg);
}