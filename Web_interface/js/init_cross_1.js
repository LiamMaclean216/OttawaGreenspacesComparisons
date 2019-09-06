"use strict";

// ###############################################################################################
// General variables
// ###############################################################################################
var image_key1;
var image_key2;
var previous_image_key1;
var previous_image_key2;
var image_path1;
var image_path2;
var previous_image_path1;
var previous_image_path2;
var back = -1;
var nb_comparisons = -1;
var recaptcha_response; 

// ###############################################################################################
// Functions
// ###############################################################################################

function captcha() {
  /**
   * Execute google reCaptcha
   */
  grecaptcha.ready(function () {
    grecaptcha.execute('', {
      action: 'homepage'
    }).then(function (token) {
      recaptcha_response = token;
    });
  });
}

function addImages() {
  var back = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : -1;

  /**
  * Changes displayed images 
  *
  * @param {number}   [back=-1] New images
  * @param {number}   [back=1] Previous images
  */
  var xhr = new XMLHttpRequest();
  xhr.addEventListener('readystatechange', function () {
    if (xhr.readyState == 4 && xhr.status == 200) {
      var response = xhr.responseText;
      console.log(response);
      var responseJSON = JSON.parse(response);
      var imagesDiv = document.getElementById('images');

      // Update general variables
      previous_image_key1 = image_key1;
      previous_image_key2 = image_key2;
      previous_image_path1 = image_path1;
      previous_image_path2 = image_path2;
      image_key1 = responseJSON.image_key1;
      image_key2 = responseJSON.image_key2;
      image_path1 = responseJSON.image_path1;
      image_path2 = responseJSON.image_path2; 

      // Remove old images
      while (imagesDiv.firstChild) {
        imagesDiv.removeChild(imagesDiv.firstChild);
      } 

      // Add left image
      var newImg = document.createElement("input");
      newImg.setAttribute('type', "image");
      newImg.setAttribute('src', image_path1);
      newImg.setAttribute('name', "winner");
      newImg.setAttribute('alt', "Image1");
      newImg.setAttribute('id', "img1");
      newImg.setAttribute('value', "left");
      imagesDiv.appendChild(newImg); 

      // Add right image
      var newImg2 = document.createElement("input");
      newImg2.setAttribute('type', "image");
      newImg2.setAttribute('src', image_path2);
      newImg2.setAttribute('name', "winner");
      newImg2.setAttribute('alt', "Image2");
      newImg2.setAttribute('id', "img2");
      newImg2.setAttribute('value', "right");
      imagesDiv.appendChild(newImg2); 

      // Change number of comparisons
      nb_comparisons -= back;
      var text_comp = document.getElementById('nb_comp');
      text_comp.innerHTML = "Number of comparisons made : " + nb_comparisons.toString();
    }
  });

  if (back == 1) {
    xhr.open('GET', 'get_images_1.php?back=' + back 
      + '&previous_image_key1=' + previous_image_key1 
      + '&previous_image_key2=' + previous_image_key2 
      + '&previous_image_path1=' + previous_image_path1 
      + '&previous_image_path2=' + previous_image_path2);
  } else {
    xhr.open('GET', 'get_images_1.php?back=' + back);
  }

  xhr.send();
}

function getIpAddress() {
  /**
  * Gets user's IP address and updates unique visitors text
  */
  var xhr = new XMLHttpRequest();
  xhr.addEventListener('readystatechange', function () {
    if (xhr.readyState == 4 && xhr.status == 200) {
      var response = xhr.responseText;
      console.log(response);
      var responseJSON = JSON.parse(response);
      var ip_address = responseJSON.ip_address; 

      // Update text  in footer
      var nb_visitor = responseJSON.nb_visitors;
      var text_visitor = document.getElementById('nb_vis');
      text_visitor.innerHTML = "Number of unique visitor : " + nb_visitor.toString();
    }
  });
  xhr.open('GET', 'get_address.php');
  xhr.send();
} 

// ###############################################################################################
// Events Functions
// ###############################################################################################


function goBack(event) {
  /**
  * Allow to go back to the 2 previous images
  */
  // Prevents to go back after initial load 
  if (previous_image_key1 && previous_image_key1 != image_key1) {
    // Set parameters to send with ajax
    image_key1 = previous_image_key1;
    image_key2 = previous_image_key2;
    image_path1 = previous_image_path1;
    image_path2 = previous_image_path2;
    back = 1;
    addImages(back);
    back = -1;
  }
  else if (previous_image_key1 == image_key1) {
    alert("Sorry, you can only go back once !") 
  }
}

function validateForm(event) {
  /**
  Updates images, variables and texts
  */
  captcha();
  event.preventDefault();
  var elemCourant = document.activeElement;
  var winner = elemCourant.value;
  var xhr = new XMLHttpRequest();
  xhr.addEventListener('readystatechange', function () {
    if (xhr.readyState == 4 && xhr.status == 200) {
      var response = xhr.responseText;
      console.log(response);
      var responseJSON = JSON.parse(response);
      var captchaMessage = responseJSON.captcha_message;

      if (captchaMessage == "Catcha ok") {
        // Update images
        addImages();
      } else {
        alert("Time has elapsed while playing the game.  Please close this box and continue…");
      }
    }
  });
  xhr.open('GET', 'write_db_1.php?' 
    + 'image_key1=' + image_key1 
    + '&image_key2=' + image_key2 
    + '&winner=' + winner 
    + '&recaptcha_response=' + recaptcha_response);
  xhr.send();
} 

// ###############################################################################################
// Initialization
// ###############################################################################################


function init() {
  /**
  Initialisation
   Initialization of captcha token
  Add the 2 initial images
  Get the IP address of the user
  Initialization of event listeners
  */
  captcha();
  addImages(back);
  getIpAddress();
  var form = document.getElementById("form");
  form.addEventListener('submit', validateForm);
  var goback = document.getElementById("goback");
  goback.addEventListener('click', goBack);
}

window.onload = init;