import smtplib


def send_approval_email(to_email,Name):
    smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
    smtp_server.starttls()
    # Replace the placeholders below with your own Gmail account credentials
    smtp_server.login('cancerdetector256@gmail.com', 'kdlzqxykcblxkmfz')

    # Compose the message
    subject = 'About the approval for Cancer Detector'
    body = f'Dear {Name}\n\nYour account has been appoved. You can now login and access the cancer detector.'
    message = f'Subject: {subject}\n\n{body}'

    # Send the email
    from_email = 'cancerdetector256@gmail.com'
    smtp_server.sendmail(from_email, to_email, message)

    # Close the SMTP server
    smtp_server.quit()

    # Close the SMTP server
    

def send_rejection_email(to_email,Name):
    smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
    smtp_server.starttls()
    # Replace the placeholders below with your own Gmail account credentials
    smtp_server.login('cancerdetector256@gmail.com', 'kdlzqxykcblxkmfz')

    # Compose the message
    subject = 'About the approval for Cancer Detector'
    body = f'Dear {Name}\nYour account has been Rejected. Re-Submit your application with correct details.'
    message = f'Subject: {subject}\n\n{body}'

    # Send the email
    from_email = 'cancerdetector256@gmail.com'
    smtp_server.sendmail(from_email, to_email, message)

    # Close the SMTP server
    smtp_server.quit()


