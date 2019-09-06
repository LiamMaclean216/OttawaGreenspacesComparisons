<!DOCTYPE html>
<html>
    <head>
        <meta charset='UTF-8'>
        <link type="text/css" rel="stylesheet" href="../css/normalize.css" />
        <link rel="stylesheet" href="../css/style.css">
        <link rel="shortcut icon" href="../css/red.png">
        <title>Winterspaces</title>
        <script src="../js/init_cross_2.js"></script>
        <script src="https://www.google.com/recaptcha/api.js?render="></script>
    </head>
    <body>
        <div id='presentation' class="round_box">
            <h1>Question 1 </h1>
            <ul><p> Choose wisely !!!!!!!!!!!.
            </p></ul>
            <ul><p><i>If you are having trouble using this application, please try on a different device. We are working on fixing some glitches. Thanks!</i></p></ul>
            <ul><a href="../index.php">Home page</a></ul>
        </div>

        <form id="form" class="round_box">
            <div id="images" class="round_box flex_around">
            
            </div>
            <div id="buttons" class="round_box flex_around" >
                <p>I have no particular preference for the right or left image.</p>
                <input class="round_button" type="submit" id="equivalent" name="winner" value="No preference">
            </div>
            <div id="gobox" class="round_box flex_around">
                <p>Go back to the previous photo pair to make a different choice</p>
                <button class="round_button" id='goback' type="button">Go Back</button>
            </div>
            
        </form>
        <div id="bottom" class="round_box flex_around">
            <div >
                <p id="nb_comp"></p>
            </div>
            <div >
                <p id="nb_vis"></p>
            </div>
        </div>
    </body>
</html>