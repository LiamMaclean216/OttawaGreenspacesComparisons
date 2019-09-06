<?php

    // ==================  CATCHA ============================
    if ($_SERVER['REQUEST_METHOD'] === 'GET' && isset($_GET['recaptcha_response'])) {
        // Build POST request:
        $recaptcha_url = 'https://www.google.com/recaptcha/api/siteverify';
        $recaptcha_secret = '';
        $recaptcha_response = $_GET['recaptcha_response'];
        // Make and decode POST request:
        $recaptcha = file_get_contents($recaptcha_url . '?secret=' . $recaptcha_secret . '&response=' . $recaptcha_response);
        $recaptcha_decode = json_decode($recaptcha);
        // Take action based on the score returned:
        if ($recaptcha_decode->success) {
            // Verified
            $captcha_message = "Catcha ok";
        } else {
            // Not verified
            $captcha_message = "Catcha not good :(";
        }
        //var_dump($recaptcha_decode) ;
    }
    // =======================================================

    // Database parameters
    $servername = "";
    $dbname = "";
    $username = "";
    $password = "";
    $options = array(PDO::MYSQL_ATTR_SSL_CA => '../ssl/BaltimoreCyberTrustRoot.crt.pem');

    // Connection to database
    try {
        
        $conn = new PDO("mysql:host=$servername;port=3306;dbname=$dbname", $username, $password, $options);
        // Set the PDO error mode to exception
        $conn->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        // echo "Connected successfully"; 
        }
    catch(PDOException $e)
        {
        // echo "Connection failed: " . $e->getMessage();
        }

    // Values received from the form
    $img_key1 = $_GET["image_key1"];
    $img_key2 = $_GET["image_key2"];
    $winner = $_GET["winner"];

    // Check values received from the form
    $message = "";
    $valid1 = preg_match('/[\w]+/', $img_key1);
    $valid2 = preg_match('/[\w]+/', $img_key2);
    $valid3 = (strlen($img_key1) == 26 && strlen($img_key2) == 26);
    $valid4 = ($winner == "right" || $winner == "left" || $winner == "No preference");

    if ($valid1 + $valid2 + $valid3 != 3){
        $message .= "Unvalid keys\n";
    }
    if ($valid4 != 1){
        $message .= "Unvalid winner\n";
    }

    // If values are ok make queries
    if ($valid1 + $valid2 + $valid3 + $valid4 == 4){
        // Get IP address of user
        function getUserIpAddr(){
            if(!empty($_SERVER['HTTP_CLIENT_IP'])){
                //ip from share internet
                $ip = $_SERVER['HTTP_CLIENT_IP'];
            }elseif(!empty($_SERVER['HTTP_X_FORWARDED_FOR'])){
                //ip pass from proxy
                $ip = $_SERVER['HTTP_X_FORWARDED_FOR'];
            }else{
                $ip = $_SERVER['REMOTE_ADDR'];
            }
            $ip_port = explode(":", $ip);
            $final_ip = $ip_port[0];
            return $final_ip;
        }
        $ip_address = getUserIpAddr();

        // Insert values inside database
        $query = $conn->prepare("INSERT INTO duels_question_4(key1, key2, winner, ip_address) VALUES(?, ?, ?, ?)");
        $query->execute(array($img_key1, $img_key2, $winner, $ip_address));

        //  Add a comparison for the user
        $query = $conn->prepare("UPDATE userswinter SET comparisons = comparisons + 1 WHERE ip_address = ?");
        $query->execute(array($ip_address));
    } else {
        echo $message;
    } 

    // Send results for debugging
    $result = array('ip_address' => $ip_address,
        'captcha_message' => $captcha_message,
        'error_message' => $message);

    echo json_encode($result);
    // End connection to database
    $conn=null

?>
