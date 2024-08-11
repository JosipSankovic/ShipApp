#include "ShipApp.h"
#include <QtWidgets/QApplication>
#include <QFile>
#include <qtextstream.h>
int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QFile f(":ShipApp/style.qss");
    if (!f.exists()) {
        qWarning("Unable to set stylesheet, file not found");
    }
    else {
        f.open(QFile::ReadOnly | QFile::Text);
        QTextStream ts(&f);
        a.setStyleSheet(ts.readAll());
    }
    ShipApp w;
    w.show();
    return a.exec();
}
