 <?php    
    /**
    * Gets user's IP address
    * @return string IP address
    */
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

    // Check if this IP address is already inside database
    $query = $conn->prepare("SELECT COUNT(*) FROM userswinter WHERE ip_address = ?");
    $query->execute([$ip_address]);
    $new_ip = $query->fetchColumn();    
    if ($new_ip == 1) {
        // Nothing to do 
    } else { 
        // Add new entry in user
        $query2 = $conn->prepare("INSERT INTO userswinter (ip_address, comparisons) VALUES( ? , 0)");
        $query2->execute([$ip_address]);
    }

    // Get number of differents visitors
    $query3 = "SELECT COUNT(*) FROM userswinter";
    $nb_visitors = $conn->query($query3)->fetchColumn();

    // Send values to ajax
    $result = array('ip_address' => $ip_address,
        'nb_visitors' => $nb_visitors);
    echo json_encode($result);

    // End connection to database
    $conn=null
?>     