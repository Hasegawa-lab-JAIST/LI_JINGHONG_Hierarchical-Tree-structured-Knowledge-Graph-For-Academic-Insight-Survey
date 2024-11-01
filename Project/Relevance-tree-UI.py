file_path = "Relevance_tree.html"

with open(file_path, "r", encoding="utf-8") as file:
    html_content = file.read()

js_code = """
    <script type="text/javascript">
    var enlargedNodes = [];
    var originalSize = 10;

    // Handle double-click event
    network.on('doubleClick', function(params) {
        var nodeId = params.nodes[0];
        if (nodeId) {
            var nodeName = nodes.get(nodeId).title.split('\\n')[0];  // Extract the first line of the label
            var pdfLink = nodes.get(nodeId).title.split('\\n').pop(); // Get the last line as the URL

            var currentText = document.getElementById('paperTitle').value;
            if (!currentText.includes(nodeName)) {
                document.getElementById('paperTitle').value = currentText + nodeName + ' ';  // Append node title on a new line
                document.getElementById('paperTitle').style.color = "purple";  // Change text color to purple
            }

            // Update URL display area to be clickable
            var urlDisplay = document.getElementById('urlDisplay');
            urlDisplay.href = pdfLink;  // Set the URL
            urlDisplay.innerText = 'See pdf'+' : '+ nodeName;  // Display "See pdf"
            urlDisplay.style.display = 'block'; // Show the link

            // Highlight the node in pink and increase its size
            network.body.data.nodes.update({ id: nodeId, color: "#FF69B4", size: originalSize * 3 });
            network.redraw();  // Redraw the network
        }
    });

    // Search bar functionality for filtering nodes by label
    function filterNodes() {
        var filterText = searchBar.value.toLowerCase();  // Get search text
        var nodesToShow = new Set();  // Create a set to store nodes that should be visible
        var edgesToShow = new Set();  // Create a set to store edges that should be visible

        // Loop over all nodes to find matches based on the label
        nodes.forEach(function(node) {
            var isNodeMatch = node.label.toLowerCase().includes(filterText);
            if (isNodeMatch) {
                nodesToShow.add(node.id);
            }
        });

        // Loop over all edges to find matches based on the edge labels or connected nodes
        edges.forEach(function(edge) {
            var isEdgeMatch = edge.label.toLowerCase().includes(filterText) || nodesToShow.has(edge.from) || nodesToShow.has(edge.to);
            
            if (isEdgeMatch) {
                // Show the matching edge
                edgesToShow.add(edge.id);
                
                // Also show the nodes connected by this edge
                nodesToShow.add(edge.from);
                nodesToShow.add(edge.to);
            }
        });

        // Update visibility of nodes based on collected sets
        nodes.forEach(function(node) {
            network.body.data.nodes.update({id: node.id, hidden: !nodesToShow.has(node.id)});
        });

        // Update visibility of edges based on collected sets
        edges.forEach(function(edge) {
            network.body.data.edges.update({id: edge.id, hidden: !edgesToShow.has(edge.id)});
        });

        // Redraw the network to apply the visibility changes
        network.redraw();
    }

    // Bind search functionality to search bar
    document.getElementById('searchBar').addEventListener('input', function() {
        filterNodes();  // Automatically filter nodes when the user types
    });

    // Handle download of the Root text area content
    document.getElementById('confirmButton').addEventListener('click', function() {
        var paperTitle = document.getElementById('paperTitle').value;
        var blob = new Blob([paperTitle], { type: 'text/plain' });
        var link = document.createElement('a');
        link.href = window.URL.createObjectURL(blob);
        link.download = 'paper-selection.txt';  // Download file as paper-selection.txt
        link.click();
    });

    // Clear the search bar and reset node visibility
    document.getElementById('clearSearchButton').addEventListener('click', function() {
        // Clear the search bar
        document.getElementById('searchBar').value = '';

        // Show all nodes
        nodes.forEach(function(node) {
            network.body.data.nodes.update({id: node.id, hidden: false});
        });

        // Show all edges
        edges.forEach(function(edge) {
            network.body.data.edges.update({id: edge.id, hidden: false});
        });

        // Redraw the network
        network.redraw();
    });

    // Reset the network state (reload page)
    document.getElementById('initButton').addEventListener('click', function() {
        window.location.reload();
    });
    </script>
"""

html_content = html_content.replace("</body>", """
    <style>
        body {
            margin: 0;
            padding: 20px;
            background-color: #f9f9f9;
        }

        .network-container {
            position: relative;
            border: 10px solid transparent;
            border-radius: 15px;
            padding: 10px;
            background-image: linear-gradient(white, white), linear-gradient(to right, #FF5733, #FFC300, #DAF7A6, #33FF57, #3375FF);
            background-origin: border-box;
            background-clip: content-box, border-box;
        }

        #network {
            border-radius: 10px;
        }

        #searchBar {
            width: 300px; 
            margin-bottom: 10px;
            border: 2px solid #97C2FC;
            border-radius: 5px;
            padding: 8px 30px 8px 40px;
            font-size: 16px;
            background-image: url('https://img.icons8.com/material-outlined/24/97C2FC/search.png');
            background-repeat: no-repeat;
            background-position: 10px center;
            background-size: 20px 20px;
        }

        textarea {
            margin-bottom: 10px;
            font-size: 18px; 
            font-weight: bold; 
            color: darkblue; 
            border: 2px solid #97C2FC;
            border-radius: 5px;
            padding: 8px;
            resize: none;
        }

        #confirmButton {
            width: 150px; 
            padding: 10px;
            background-color: #97C2FC;
            color: white;
            font-size: 16px;
            font-weight: bold;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
            margin-bottom: 15px;
        }

        #confirmButton:hover {
            background-color: #6ea2d8;
        }

        #initButton {
            width: 50px; 
            height: 30px; 
            margin-left: 5px; 
            background-color: #33FF57;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: background-color 0.3s;
        }

        #initButton:hover {
            background-color: #29cc47;
        }

        .inheritance-title {
            font-family: 'Arial Rounded MT Bold', sans-serif;
            font-size: 30px;
            margin: 0 10px;
            align-self: center;
        }
        
        #clearSearchButton {
            width: 50px; 
            height: 30px; 
            margin-left: 5px; 
            background-color: #FF5733;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: background-color 0.3s;
        }

        #clearSearchButton:hover {
            background-color: #d14b2b;
        }

        .flex-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 5px;
        }

        .right-buttons {
            display: flex;
            gap: 5px;
            align-items: center;
        }
    </style>
    <div class="network-container">
        <div id="network"></div>
    </div>
    
    <div style="margin-top: 20px; display: flex; flex-direction: column; align-items: stretch;">
        <div class="flex-container">
            <div class="left-buttons">
                <input type="text" id="searchBar" placeholder="Enter the keyword">
                <button id="clearSearchButton">Clear</button>
                <button id="initButton">Init</button>
            </div>
            <span class="inheritance-title">
                <a id="urlDisplay" href="#" target="_blank" style="color: blue; text-decoration: underline; display: none;">See pdf</a>
            </span>
            <div class="right-buttons">
                <button id="confirmButton">Confirm selected paper</button>
            </div>
        </div>
        <textarea id="paperTitle" rows="2" cols="20">Paper: </textarea>
    </div>
""" + js_code + "</body>")

output_file_path = "Relevance_tree_UI.html"
with open(output_file_path, "w", encoding="utf-8") as file:
    file.write(html_content)
