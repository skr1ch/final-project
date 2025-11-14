import sqlite3
from datetime import datetime, timedelta

def get_db_connection():
    conn = sqlite3.connect('wardrobe.db')
    conn.row_factory = sqlite3.Row
    return conn

def get_unread_notifications_count(user_id):
    """Get count of unread notifications for a user"""
    conn = get_db_connection()
    count = conn.execute('SELECT COUNT(*) FROM notifications WHERE user_id = ? AND is_read = 0', (user_id,)).fetchone()[0]
    conn.close()
    return count

def get_user_notifications(user_id, limit=20):
    """Get all notifications for a user"""
    conn = get_db_connection()
    notifications = conn.execute('''
        SELECT n.*, c.cloth_name, c.image_path 
        FROM notifications n 
        LEFT JOIN clothes c ON n.cloth_id = c.id 
        WHERE n.user_id = ? 
        ORDER BY n.created_at DESC 
        LIMIT ?
    ''', (user_id, limit)).fetchall()
    conn.close()
    return notifications

def mark_notification_read(notification_id, user_id):
    """Mark a notification as read"""
    conn = get_db_connection()
    conn.execute('UPDATE notifications SET is_read = 1 WHERE id = ? AND user_id = ?', (notification_id, user_id))
    conn.commit()
    conn.close()

def mark_all_notifications_read(user_id):
    """Mark all notifications as read for a user"""
    conn = get_db_connection()
    conn.execute('UPDATE notifications SET is_read = 1 WHERE user_id = ?', (user_id,))
    conn.commit()
    conn.close()

def create_laundry_reminder(user_id, cloth_id, message):
    """Create a laundry reminder notification"""
    conn = get_db_connection()
    conn.execute('''
        INSERT INTO notifications (user_id, cloth_id, message, type, is_read)
        VALUES (?, ?, ?, 'laundry_reminder', 0)
    ''', (user_id, cloth_id, message))
    conn.commit()
    conn.close()

def check_daily_laundry_reminders():
    """Check and create daily laundry reminders for all users"""
    conn = get_db_connection()
    
    # Get all users
    users = conn.execute('SELECT id FROM users').fetchall()
    
    for user in users:
        user_id = user['id']
        
        # Get clothes that need washing soon
        clothes = conn.execute('''
            SELECT c.*, 
                   julianday('now') - julianday(c.last_worn) as days_since_worn
            FROM clothes c
            WHERE c.user_id = ? AND c.last_worn IS NOT NULL
        ''', (user_id,)).fetchall()
        
        for cloth in clothes:
            days_since_worn = cloth['days_since_worn'] or 0
            
            # Predict wash time
            from app import predict_wash_time_ai
            wash_due_days = predict_wash_time_ai(
                fabric_type=cloth['fabric_type'],
                usage_count=cloth['wear_count'],
                days_since_last_wash=days_since_worn,
                temperature_avg=25,
                humidity_avg=60,
                care_instructions=cloth['care_instructions']
            )
            
            # Create notification if wash is due today or tomorrow
            if wash_due_days <= 1:
                message = f"🚨 Time to wash {cloth['cloth_name']}! It's been {int(days_since_worn)} days since last wash."
                create_laundry_reminder(user_id, cloth['id'], message)
            elif wash_due_days <= 3:
                message = f"⚠️ {cloth['cloth_name']} will need washing in {wash_due_days} days."
                create_laundry_reminder(user_id, cloth['id'], message)
    
    conn.close()