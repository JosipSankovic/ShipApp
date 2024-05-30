#include "ShipApp.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    ShipApp w;
    w.show();
    return a.exec();
}
