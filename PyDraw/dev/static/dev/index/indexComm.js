console.log(">> In index save");

function getData()
{
    var imgData = canvas.toDataURL();
    return imgData;
}


$( document ).ready(function() {
    $( "#sendBtn" ).click(function() {
//  alert( "Handler for .click() called." );
console.log(OPTION);
console.log('Click');
imgData = getData()
$.ajax({url: '/dev/process',
        data:{ info : imgData},
        success: function(result){
        console.log(result);

        $("#right").append(result.info);

//        var right = document.getElementById("right")
//        var image = document.createElement("img");
//        image.src = result._1;
//        right.appendChild(image);
        if (OPTION != 'NULL')
        {
             plotImages(result, OPTION);
        }

    }});
});
});


function plotImages(result, OPTION)
{
    var keys = Object.keys(result);
    var values = Object.values(result)

    var container = document.getElementById("right")

    if (OPTION == 'words') {
        for (var i = 0; i < keys.length; i++) {
            var word_cont = document.createElement("div");

            console.log("Iteratia : " + i);

            if (keys[i].length == 4) {
                var word = document.createElement("img");
                word.className = "word"
                word.src = result[keys[i]];
//                var text = document.createTextNode(obj[keys[i]]); // astea doua linii sunt inlocuite de src
//                word.appendChild(text);
                word_cont.appendChild(word)

                console.log(i + "  " + keys[i]);

                ok = 0;
                i++;
                while (keys[i].length == 6) {
                    ok = 1;
                    console.log(">>" + i + " " + keys[i]);
                    var char = document.createElement("img");
                    char.className = "char";
//                    var text = document.createTextNode(obj[keys[i]])
//                    char.appendChild(text);
                    char.src = result[keys[i]];
                    word_cont.appendChild(char);

                    if (typeof keys[i + 1] === 'undefined') {
                        console.log("Urmatorul nu exista");
                        ok = 0;
                        break;
                    }
                    else {
                        i++;
                    }
                }
                console.log("Inainte de ok : " + i);
                if (ok == 1) { i--; }
                console.log("Dupa ok : " + i);
            }

            container.appendChild(word_cont);
        }
    }


    if (OPTION == 'lines') {
        for (var i = 0; i < keys.length; i++) {
            var line_cont = document.createElement("div");

            console.log("Iteratia : " + i);

            if (keys[i].length == 2) {
                var line = document.createElement("img");
                line.className = "line"
//                var text = document.createTextNode(obj[keys[i]]); // astea doua linii sunt inlocuite de src
//                line.appendChild(text);
                line.src = result[keys[i]];
                line_cont.appendChild(line);

                console.log(i + "  " + keys[i]);

                ok = 0;
                i++;
                while (i < keys.length) {
                    if (keys[i].length == 4) {
                        ok = 1;
                        console.log(">>" + i + " " + keys[i]);
                        var word = document.createElement("img");
                        word.className = "word";
//                        var text = document.createTextNode(obj[keys[i]])
//                        word.appendChild(text);
                        word.src = result[keys[i]];
                        line_cont.appendChild(word);
                    }

                    if (typeof keys[i + 1] === 'undefined') {
                            console.log("Urmatorul nu exista");
                            ok = 0;
                            break;
                        }
                        else {
                            i++;
                        }
                }
                console.log("Inainte de ok : " + i);
                if (ok == 1) { i--; }
                console.log("Dupa ok : " + i);
            }

            container.appendChild(line_cont);
        }
    }
}