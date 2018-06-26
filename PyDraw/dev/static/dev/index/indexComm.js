console.log(">> In index save");
var text4Speech = "";
function getData() {
    var imgData = canvas.toDataURL();
    return imgData;
}


$(document).ready(function () {
    $("#sendBtn").click(function () {
        console.log(OPTION);
        console.log('Click');
        imgData = getData()
        $.ajax({
            url: '/dev/process',
            data: { info: imgData },
            success: function (result) {
                console.log(result);
                destroy_elements();
                //        $("#right").append(result.info);

                if (OPTION != 'NULL') {
                    plotImages(result, OPTION);
                }

            }
        });
    });
});


function plotImages(obj, OPTION) {
    var keys = Object.keys(obj);
    var source = document.getElementById('content');


    if (OPTION == "lines") {

        for (var i = 0; i < keys.length; i++) {
            if (keys[i].length == 2) {

                // document.write(keys[i] + "  :  " + (keys[i].split('_')));
                // document.write("<br/>");
                var line_cont = document.createElement("div");
                line_cont.className = "lineContainer";

                var line = document.createElement("img");
                line.className = "line";
                // var text = document.createTextNode(obj[keys[i]]);
                // line.appendChild(text);
                line.src = obj[keys[i]];
                // document.write(keys[i] + "<br/>");
                line_cont.appendChild(line);
                for (var j = 0; j < keys.length; j++) {
                    if (keys[j].length == 4) {
                        ks = keys[j].split('_');
                        // document.write(keys[j] + " : linia " + ks[1]);
                        if (ks[1] == keys[i][1]) {
                            // document.write(keys[j]);
                            // document.write("<br/>")

                            var word = document.createElement("img");
                            word.className = "word";
                            word.src = obj[keys[j]];
                            // var text = document.createTextNode(obj[keys[j]]);
                            // word.appendChild(text);

                            line_cont.appendChild(word);
                        }
                    }
                }
                // document.write("<br/>");
                source.appendChild(line_cont);
            }
        }
    }

    else {
        text4Speech = "";
        // aici adaig buton audio si care va apela o metoda de mai jos
        var res_div = document.getElementById('audio_div');
        var audioBtn = document.createElement('div');
        var textAudioBtn = document.createTextNode("Audio");
        var audioImg = document.createElement("img");
        // audioImg.src = 
        audioBtn.className = "audioBtn";
        audioBtn.appendChild(textAudioBtn);
        audioBtn.addEventListener('click', getAudio);
        res_div.appendChild(audioBtn);

        for (var i = 0; i < keys.length; i++) {
            if (keys[i].length == 4) {

                // document.write(keys[i] + "  :  " + (keys[i].split('_')));
                // document.write("<br/>");
                var word_cont = document.createElement("div");
                word_cont.className = "wordContainer";
                text4Speech = text4Speech + "  ";
                var word = document.createElement("img");
                word.className = "wordbase";
                // var text = document.createTextNode(obj[keys[i]]);
                // line.appendChild(text);
                word.src = obj[keys[i]];
                // document.write(keys[i] + "<br/>");
                word_cont.appendChild(word);
                for (var j = 0; j < keys.length; j++) {
                    if (keys[j].length == 6) {
                        // ks = keys[j].split('_');
                        char_word_sig = keys[j].slice(0, 4);//signature of the word part in char
                        word_sig = keys[i];
                        // document.write(keys[j] + " : linia " + ks[1]);
                        //nu cred ca e sufcient sa comparam doar cu elementul[1],
                        //trebuie comparat indice cuvant cu cuvant, si indice line cu line
                        //asta era inainte : ks[1] == keys[i][1]
                        if (char_word_sig == word_sig) {
                            // document.write(keys[j]);
                            // document.write("<br/>")
                            var char_cont = document.createElement("div")
                            char_cont.className = "char_cont";
                            var char = document.createElement("img");
                            var recieved_str = obj[keys[j]]
                            var label = document.createTextNode(recieved_str[0]);
                            text4Speech = text4Speech + recieved_str[0];
                            // var label = document.createTextNode("NaC"); //AICI AM MODIFICAT
                            var label_wrapper = document.createElement("div");
                            label_wrapper.appendChild(label);
                            label_wrapper.className="label";
                            char.className = "char";
                            // char.src = obj[keys[j]]
                            char.src = recieved_str.slice(1, recieved_str.len);
                            char_cont.appendChild(char);
                            char_cont.appendChild(label_wrapper);
                            // var text = document.createTextNode(obj[keys[j]]);
                            // word.appendChild(text);

                            word_cont.appendChild(char_cont);
                        }
                    }
                }
                // document.write("<br/>");
                source.appendChild(word_cont);
            }

        }
        // var text4sp = document.createTextNode(text4Speech);
        // source.appendChild(text4sp);
    }
}

function getAudio()
{
    console.log("get Audio accesed");
    var msg = new SpeechSynthesisUtterance(text4Speech);
            window.speechSynthesis.speak(msg);

}