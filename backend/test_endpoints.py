import unittest
import json
import unittest
import json
from app import app

class TestEndpoints(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.client = self.app.test_client()
        self.app.config['TESTING'] = True

    def test_health_check(self):
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['status'], 'healthy')

    def test_analyze_vehicle_fit(self):
        data = {
            'vehicle_type': 'truck',
            'route_points': [
                {'lat': 51.505, 'lng': -0.09},
                {'lat': 51.51, 'lng': -0.1}
            ]
        }
        response = self.client.post('/analyze_vehicle_fit', 
                                  data=json.dumps(data),
                                  content_type='application/json')
        if response.status_code != 200:
            print(f"\n❌ analyze_vehicle_fit failed: {response.status_code}")
            print(response.data.decode('utf-8'))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json['success'])
        self.assertIn('fits', response.json)

    def test_generate_alternative_route(self):
        data = {
            'locations': [
                {'lat': 51.505, 'lng': -0.09},
                {'lat': 51.51, 'lng': -0.1},
                {'lat': 51.52, 'lng': -0.12}
            ],
            'transport': 'driving'
        }
        response = self.client.post('/generate_alternative_route',
                                  data=json.dumps(data),
                                  content_type='application/json')
        if response.status_code != 200:
            print(f"\n❌ generate_alternative_route failed: {response.status_code}")
            print(response.data.decode('utf-8'))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json['success'])
        self.assertIn('alternative_routes', response.json)

if __name__ == '__main__':
    unittest.main()
