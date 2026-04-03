1. Federated Learning Infrastructure

    Server (server_terminal.py) - Aggregates model updates

    Clients (client_terminal.py) - Train locally on each bank's data

    Real-time communication - Banks send only model weights, not raw data

2. Feature Alignment (The Secret Sauce)

Created 35 shared features that both banks understand:

    velocity_signal = Transactions/hour (Bank B) = Activity frequency (Bank A)

    geographic_risk = Distance to merchant (Bank B) = Foreign request (Bank A)

    time_anomaly = Night transactions (Bank B) = Days since request (Bank A)

3. Class Imbalance Handling

    70/30 sampling - Each batch has 70% fraud, 30% legitimate

    Forces model to learn fraud patterns despite only 1% fraud in real data

4. Real-Time Detection System

    Web app (app.py) - Test transactions visually

    Live demo (realtime_demo.py) - Simulates transaction stream

    Fraud types (fraud_types.py) - Categorizes detected fraud
